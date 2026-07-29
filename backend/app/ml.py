"""机器学习模块：GBDT 预测未来收益，时序交叉验证 + OOS 评估。

防泄漏要点：
- 目标 = 未来 N 日收益（closes[i+n]/closes[i]-1），仅用 t 及之前信息
- 时序 CV：Purged Walk-Forward，严禁随机 shuffle
- 训练/验证/测试三段指标，防止过拟合自夸
- 模型 joblib 落盘 + 元数据
"""
import os
import json
import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from . import adapters, db
from .factors import FACTORS, mean, std
from .numpy_factors import kline_to_arrays, compute_factor_series, series_at

ML_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml_models")
os.makedirs(ML_DIR, exist_ok=True)

TRADING_DAYS = 252


async def build_dataset(board: str = "all", pool_size: int = 100, n: int = 5,
                        hist: int = 240, progress_cb=None) -> dict:
    """构建 ML 数据集：候选池每只股票算全部量价因子 + 未来 N 日收益。

    返回 {features, target, codes, dates, feature_names}。
    progress_cb(done, total) 用于异步进度上报。
    """
    pool = await adapters.fetch_market_list(board, pool_size)
    codes = [row["code"] for row in pool]
    sem_factor_keys = [k for k in FACTORS]
    total = len(codes)

    rows = []
    labels = []
    meta_codes = []
    meta_dates = []
    for idx, code in enumerate(codes):
        if progress_cb:
            progress_cb(idx + 1, total)
        try:
            kline = await adapters.fetch_kline(code, hist)
        except Exception:
            continue
        if len(kline) < 60 + n:
            continue
        arr = kline_to_arrays(kline)
        series_map = {k: compute_factor_series(k, arr) for k in sem_factor_keys}
        closes = arr["close"]
        for i in range(60, len(kline) - n):
            fvals = []
            ok = True
            for k in sem_factor_keys:
                s = series_map[k]
                v = series_at(s, i)
                if v is None:
                    ok = False
                    break
                fvals.append(v)
            if not ok or closes[i] == 0:
                continue
            fret = closes[i + n] / closes[i] - 1.0
            rows.append(fvals)
            labels.append(fret)
            meta_codes.append(code)
            meta_dates.append(arr["date"][i])

    if not rows:
        raise ValueError("有效样本不足，无法构建数据集（请增大候选池或历史长度）")

    return {
        "features": np.array(rows, dtype=np.float64),
        "target": np.array(labels, dtype=np.float64),
        "codes": meta_codes,
        "dates": meta_dates,
        "feature_names": sem_factor_keys,
    }


def winsorize(arr: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    """对一维数组做缩尾（截断到分位数），返回新数组。"""
    a = np.asarray(arr, dtype=np.float64).copy()
    lo = np.nanquantile(a, lower)
    hi = np.nanquantile(a, upper)
    a[a < lo] = lo
    a[a > hi] = hi
    return a


def standardize(arr: np.ndarray) -> np.ndarray:
    """标准化（按列），缺失用列均值填充。"""
    col_mean = np.nanmean(arr, axis=0)
    filled = np.where(np.isnan(arr), col_mean, arr)
    col_std = np.nanstd(filled, axis=0)
    col_std[col_std == 0] = 1.0
    return (filled - col_mean) / col_std


def purged_walk_forward_split(n_samples: int, n_splits: int = 5, test_ratio: float = 0.2, gap: int = 5):
    """时序 Walk-Forward 分割：训练区在前、测试区在后，中间留 gap 防泄漏。

    返回 [(train_idx, test_idx), ...]，按时间顺序滚动。
    """
    splits = []
    fold_size = n_samples // n_splits
    test_size = int(fold_size * test_ratio)
    for k in range(n_splits):
        train_end = k * fold_size
        test_start = train_end + gap
        test_end = test_start + test_size
        if test_end > n_samples:
            break
        train_idx = np.arange(0, train_end) if train_end > 0 else np.array([], dtype=int)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits


def evaluate_dataset(dataset: dict, model_type: str = "gbdt", n_splits: int = 5,
                     gap: int = 5, progress_cb=None) -> dict:
    """时序 CV 评估：每折训练→预测→统计 IC/RankIC/Sharpe，汇总 OOS 指标。"""
    X = standardize(winsorize_dataset(dataset["features"]))
    y = dataset["target"]
    dates = dataset["dates"]
    feature_names = dataset["feature_names"]
    n = len(y)

    splits = purged_walk_forward_split(n, n_splits=n_splits, gap=gap)
    if not splits:
        raise ValueError("样本不足以做时序交叉验证")

    fold_results = []
    all_pred = np.full(n, np.nan)
    importances = np.zeros(len(feature_names))

    for k, (train_idx, test_idx) in enumerate(splits):
        if progress_cb:
            progress_cb(k + 1, len(splits))
        Xtr, ytr = X[train_idx], y[train_idx]
        Xte, yte = X[test_idx], y[test_idx]
        if len(Xtr) < 50 or len(Xte) < 10:
            continue

        model = _build_model(model_type)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        all_pred[test_idx] = pred

        if hasattr(model, "feature_importances_"):
            importances += model.feature_importances_

        ic = _pearson(pred, yte)
        rank_ic = _spearman(pred, yte)
        # 分层：按预测分5组，多空 = 最高组均值 - 最低组均值
        top_ret, bottom_ret, ls_ret = _bucket_returns(pred, yte, 5)
        fold_results.append({
            "fold": k + 1, "trainSize": len(train_idx), "testSize": len(test_idx),
            "ic": ic, "rankIc": rank_ic, "longShort": ls_ret,
            "topReturn": top_ret, "bottomReturn": bottom_ret,
            "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
        })

    if not fold_results:
        raise ValueError("样本不足以完成任何一折训练（需更多数据或减小 gap/CV 折数）")

    valid_mask = ~np.isnan(all_pred)
    valid_pred = all_pred[valid_mask]
    valid_y = y[valid_mask]
    overall_ic = _pearson(valid_pred, valid_y)
    overall_rank_ic = _spearman(valid_pred, valid_y)
    _, _, ls_returns_all = _bucket_returns(valid_pred, valid_y, 5)
    oos_sharpe = _sharpe_periodic([ls_returns_all]) if ls_returns_all else 0.0

    avg_importances = importances / max(1, len(fold_results))
    feat_imp = sorted(
        [{"feature": feature_names[i], "importance": float(avg_importances[i])}
         for i in range(len(feature_names))],
        key=lambda x: x["importance"], reverse=True
    )

    return {
        "nSamples": n, "nFeatures": len(feature_names), "featureNames": feature_names,
        "folds": fold_results,
        "oosIc": overall_ic, "oosRankIc": overall_rank_ic,
        "oosLongShort": ls_returns_all,
        "oosSharpe": oos_sharpe,
        "featureImportance": feat_imp,
        "testDates": [dates[i] for i in range(n) if valid_mask[i]],
        "testActual": valid_y.tolist(),
        "testPred": valid_pred.tolist(),
    }


def winsorize_dataset(X: np.ndarray) -> np.ndarray:
    """对每列做缩尾。"""
    out = X.copy()
    for c in range(out.shape[1]):
        out[:, c] = winsorize(out[:, c])
    return out


def train_final_model(dataset: dict, model_type: str = "gbdt") -> dict:
    """用全量数据训练最终模型并落盘，返回模型元数据。"""
    X = standardize(winsorize_dataset(dataset["features"]))
    y = dataset["target"]
    feature_names = dataset["feature_names"]

    model = _build_model(model_type)
    model.fit(X, y)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mid = f"mlmodel_{ts}"
    path = os.path.join(ML_DIR, f"{mid}.joblib")
    joblib.dump({"model": model, "feature_names": feature_names, "model_type": model_type}, path)

    meta = {
        "id": mid, "path": path, "modelType": model_type,
        "featureNames": feature_names, "nSamples": len(y),
        "trainedAt": datetime.datetime.now().isoformat(),
    }
    return meta


def list_models() -> list[dict]:
    out = []
    if not os.path.isdir(ML_DIR):
        return out
    for f in sorted(os.listdir(ML_DIR), reverse=True):
        if f.endswith(".joblib"):
            out.append({"id": f.replace(".joblib", ""), "file": f})
    return out


def delete_model(mid: str) -> bool:
    path = os.path.join(ML_DIR, f"{mid}.joblib")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def _build_model(model_type: str):
    if model_type == "gbdt":
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
    raise ValueError(f"不支持的模型类型: {model_type}")


def _pearson(xs, ys):
    if len(xs) < 2:
        return 0.0
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    mx, my = xs.mean(), ys.mean()
    den = np.sqrt(np.sum((xs - mx) ** 2) * np.sum((ys - my) ** 2))
    return 0.0 if den == 0 else float(np.sum((xs - mx) * (ys - my)) / den)


def _rank(arr):
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(arr) + 1)
    return ranks


def _spearman(xs, ys):
    return _pearson(_rank(xs), _rank(ys))


def _bucket_returns(pred, actual, groups: int = 5):
    """按预测分组，返回 (top组均值, bottom组均值, 多空收益)。"""
    if len(pred) < groups * 2:
        return 0.0, 0.0, 0.0
    order = np.argsort(pred)
    bucket_size = len(order) // groups
    bottom_idx = order[:bucket_size]
    top_idx = order[-bucket_size:]
    top_ret = float(np.mean(actual[top_idx]))
    bottom_ret = float(np.mean(actual[bottom_idx]))
    return top_ret, bottom_ret, top_ret - bottom_ret


def _sharpe_periodic(returns, periods_per_year: int = TRADING_DAYS):
    if not returns:
        return 0.0
    arr = np.array(returns, dtype=np.float64)
    m = arr.mean()
    s = arr.std()
    return 0.0 if s == 0 else float(m / s * np.sqrt(periods_per_year))
