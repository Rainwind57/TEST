"""机器学习模块：GBDT 预测未来收益，时序交叉验证 + OOS 评估。

防泄漏要点：
- 目标 = 未来 N 日收益（closes[i+n]/closes[i]-1），仅用 t 及之前信息
- 时序 CV：Purged Walk-Forward，严禁随机 shuffle
- 训练/验证/测试三段指标，防止过拟合自夸
- 模型 joblib 落盘 + 元数据
"""
import os
import json
import asyncio
import datetime
from collections import defaultdict

import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from . import adapters, db
from .factors import (
    FACTORS, SNAPSHOT_FACTORS, mean, std, bucket_index,
    annualized_return, annualized_volatility, sharpe_ratio, sortino_ratio,
    max_drawdown, calmar_ratio, win_rate, information_coefficient_stats,
    round_trip_cost_rate, snapshot_factor_value,
)
from .numpy_factors import kline_to_arrays, compute_factor_series, series_at

ML_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml_models")
os.makedirs(ML_DIR, exist_ok=True)

TRADING_DAYS = 252


async def build_dataset(board: str = "all", pool_size: int = 100, n: int = 5,
                        hist: int = 240, progress_cb=None,
                        use_snapshot: bool = False) -> dict:
    """构建 ML 数据集：候选池每只股票算全部量价因子 + 未来 N 日收益。

    返回 {features, target, codes, dates, feature_names}。
    use_snapshot=True 时追加 pe/pb/turnover 最新快照作为静态特征（含前视风险，
    仅探索用；推理 score_latest/backtest_model 会按 feature_names 一致拉取）。
    """
    pool = await adapters.fetch_market_list(board, pool_size)
    codes = [row["code"] for row in pool]
    sem_factor_keys = [k for k in FACTORS]
    snapshot_keys = ["pe", "pb", "turnover"] if use_snapshot else []
    row_by_code = {r["code"]: r for r in pool}
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
            # 快照特征（最新值作静态特征，含前视风险，仅探索用）
            if use_snapshot:
                r = row_by_code.get(code)
                if not r:
                    continue
                sv = [snapshot_factor_value(r, k) for k in snapshot_keys]
                if any(v is None for v in sv):
                    continue
                fvals.extend(sv)
            fret = closes[i + n] / closes[i] - 1.0
            rows.append(fvals)
            labels.append(fret)
            meta_codes.append(code)
            meta_dates.append(arr["date"][i])

    if not rows:
        raise ValueError("有效样本不足，无法构建数据集（请增大候选池或历史长度）")

    # 按 (date, code) 排序后再切分，杜绝时序 CV 训练集混入晚于测试集的日期样本。
    order = sorted(range(len(rows)), key=lambda k: (meta_dates[k], meta_codes[k]))
    rows = [rows[k] for k in order]
    labels = [labels[k] for k in order]
    meta_codes = [meta_codes[k] for k in order]
    meta_dates = [meta_dates[k] for k in order]

    return {
        "features": np.array(rows, dtype=np.float64),
        "target": np.array(labels, dtype=np.float64),
        "codes": meta_codes,
        "dates": meta_dates,
        "feature_names": sem_factor_keys + snapshot_keys,
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


def _fit_preprocess(X: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> dict:
    """在训练集上拟合缩尾分位数 + 标准化均值/标准差，返回可复用参数。

    用于时序 CV 每折内 fit、再 apply 到测试折，杜绝「全样本预处理后再切分」的泄漏。
    """
    lo = np.nanquantile(X, lower, axis=0)
    hi = np.nanquantile(X, upper, axis=0)
    Xc = np.clip(X, lo, hi)  # NaN 经 clip 仍为 NaN
    col_mean = np.nanmean(Xc, axis=0)
    Xf = np.where(np.isnan(Xc), col_mean, Xc)
    col_std = np.nanstd(Xf, axis=0)
    col_std[col_std == 0] = 1.0
    return {"lo": lo, "hi": hi, "mean": col_mean, "std": col_std}


def _apply_preprocess(X: np.ndarray, params: dict) -> np.ndarray:
    """用已拟合参数对新数据缩尾 + 标准化（NaN 用训练均值填充）。"""
    Xc = np.where(np.isnan(X), params["mean"], np.clip(X, params["lo"], params["hi"]))
    return (Xc - params["mean"]) / params["std"]


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
    """时序 CV 评估：每折在训练折内 fit 预处理（缩尾+标准化）再 apply 到测试折，
    杜绝预处理泄漏；OOS Sharpe 按调仓日聚合多空序列计算（旧版单元素列表恒为 0）。"""
    X_raw = dataset["features"]
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
    oos_records = []  # (date, pred, actual) 按日聚合算 OOS Sharpe

    for k, (train_idx, test_idx) in enumerate(splits):
        if progress_cb:
            progress_cb(k + 1, len(splits))
        if len(train_idx) < 50 or len(test_idx) < 10:
            continue
        params = _fit_preprocess(X_raw[train_idx])
        Xtr = _apply_preprocess(X_raw[train_idx], params)
        Xte = _apply_preprocess(X_raw[test_idx], params)
        ytr, yte = y[train_idx], y[test_idx]

        model = _build_model(model_type)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        all_pred[test_idx] = pred
        for j, idx in enumerate(test_idx):
            oos_records.append((dates[idx], float(pred[j]), float(yte[j])))

        if hasattr(model, "feature_importances_"):
            importances += model.feature_importances_

        ic = _pearson(pred, yte)
        rank_ic = _spearman(pred, yte)
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
    oos_sharpe = _oos_sharpe_by_date(oos_records)

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
    """用全量数据训练最终模型并落盘，返回模型元数据。

    预处理参数（缩尾分位数 + 均值 + 标准差）随 joblib 一起落盘 + sidecar JSON 元数据，
    推理时复现预处理（旧版只存 model+feature_names，推理无法复现缩尾/标准化）。
    """
    X_raw = dataset["features"]
    y = dataset["target"]
    feature_names = dataset["feature_names"]

    params = _fit_preprocess(X_raw)
    X = _apply_preprocess(X_raw, params)
    model = _build_model(model_type)
    model.fit(X, y)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mid = f"mlmodel_{ts}"
    path = os.path.join(ML_DIR, f"{mid}.joblib")
    joblib.dump({
        "model": model,
        "feature_names": feature_names,
        "model_type": model_type,
        "preprocess": params,
    }, path)

    meta = {
        "id": mid, "path": path, "modelType": model_type,
        "featureNames": feature_names, "nSamples": len(y),
        "trainedAt": datetime.datetime.now().isoformat(),
    }
    try:
        with open(os.path.join(ML_DIR, f"{mid}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
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


def _oos_sharpe_by_date(oos_records: list, groups: int = 5) -> float:
    """按调仓日聚合 OOS 预测与真实收益，每日算最高组-最低组多空收益，
    形成日频时序后年化夏普（修复旧版对单标量求 std 恒为 0 的 bug）。"""
    if not oos_records:
        return 0.0
    by_date = defaultdict(list)
    for d, p, a in oos_records:
        by_date[d].append((p, a))
    ls_series = []
    for d in sorted(by_date):
        items = by_date[d]
        if len(items) < groups * 2:
            continue
        preds = np.array([x[0] for x in items])
        acts = np.array([x[1] for x in items])
        order = np.argsort(preds)
        k = len(order) // groups
        if k < 1:
            continue
        top = float(np.mean(acts[order[-k:]]))
        bot = float(np.mean(acts[order[:k]]))
        ls_series.append(top - bot)
    return _sharpe_periodic(ls_series) if ls_series else 0.0


# ---------------- 模型推理：截面打分 + ML 信号分层回测 ----------------

BENCHMARKS = {
    "none": None,
    "hs300": "sh000300",
    "zz500": "sh000905",
    "sse": "sh000001",
}


def _price_limit_ratio(code: str, is_st: bool) -> float:
    """A 股涨跌停幅度：创业板(sz30)/科创板(sh68)±20%、北交所(bj)±30%、ST±5%、主板±10%。"""
    if is_st:
        return 0.05
    if code.startswith("sz30") or code.startswith("sh68"):
        return 0.20
    if code.startswith("bj"):
        return 0.30
    return 0.10


def _load_model(mid: str) -> dict:
    """加载落盘模型 bundle（含预处理参数）。"""
    path = os.path.join(ML_DIR, f"{mid}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型不存在: {mid}")
    bundle = joblib.load(path)
    if "preprocess" not in bundle:
        raise ValueError(f"模型 {mid} 缺少预处理参数，无法推理（请用修复版重新训练）")
    return bundle


async def score_latest(mid: str, board: str = "all", pool_size: int = 100,
                       progress_cb=None) -> list[dict]:
    """加载落盘模型对候选池最新截面打分，返回按预测分降序的打分列表。"""
    bundle = _load_model(mid)
    feature_names = bundle["feature_names"]
    model = bundle["model"]
    params = bundle["preprocess"]
    snap_set = set(SNAPSHOT_FACTORS)

    pool = await adapters.fetch_market_list(board, pool_size)
    sem = asyncio.Semaphore(15)
    collected = []

    async def one(row):
        code = row["code"]
        async with sem:
            try:
                kline = await adapters.fetch_kline(code, 260)
            except Exception:
                return
        if len(kline) < 60:
            return
        arr = kline_to_arrays(kline)
        i = len(kline) - 1
        fvals = []
        for k in feature_names:
            if k in snap_set:
                v = snapshot_factor_value(row, k)  # 快照因子取最新行情字段
            else:
                v = series_at(compute_factor_series(k, arr), i)
            if v is None:
                return
            fvals.append(v)
        collected.append({"code": code, "name": row.get("name", ""), "fvals": fvals})
        if progress_cb:
            progress_cb(len(collected), len(pool))

    await asyncio.gather(*(one(r) for r in pool))
    if not collected:
        raise ValueError("无可用样本（候选池股票历史不足或因子缺失）")

    X = np.array([r["fvals"] for r in collected], dtype=np.float64)
    Xp = _apply_preprocess(X, params)
    preds = model.predict(Xp)
    rows = [{"code": r["code"], "name": r["name"], "score": float(p)}
            for r, p in zip(collected, preds)]
    rows.sort(key=lambda x: x["score"], reverse=True)
    for rank, r in enumerate(rows, start=1):
        r["rank"] = rank
    return rows


async def backtest_model(mid: str, board: str = "all", pool_size: int = 60, groups: int = 5,
                         n: int = 5, hist: int = 180, commission_rate: float = 0.00025,
                         stamp_duty: float = 0.001, slippage: float = 0.001,
                         benchmark: str = "none", apply_cost: bool = True,
                         progress_cb=None) -> dict:
    """用落盘模型预测分作为截面信号做分层回测，响应结构与 /api/select/backtest 一致，
    前端图表零成本复用。每个调仓日对每只股票取当日因子特征→模型预测分→分层。"""
    bundle = _load_model(mid)
    feature_names = bundle["feature_names"]
    model = bundle["model"]
    params = bundle["preprocess"]
    snap_set = set(SNAPSHOT_FACTORS)
    tech_keys = [k for k in feature_names if k not in snap_set]
    snap_keys = [k for k in feature_names if k in snap_set]

    groups = max(2, min(10, groups))
    n = max(1, n)
    hist = max(60, hist)

    bench_code = BENCHMARKS.get(benchmark)
    if benchmark != "none" and bench_code is None:
        raise ValueError(f"未知基准: {benchmark}")

    pool = await adapters.fetch_market_list(board, pool_size)
    codes = [row["code"] for row in pool]
    is_st = {r["code"]: ("ST" in r.get("name", "") or "*ST" in r.get("name", ""))
             for r in pool}
    sem = asyncio.Semaphore(15)

    async def fetch_one(code):
        async with sem:
            try:
                kline = await adapters.fetch_kline(code, hist + n + 25)
            except Exception:
                return code, []
            return code, kline

    fetched = await asyncio.gather(*(fetch_one(c) for c in codes))
    series = {code: kl for code, kl in fetched if len(kl) >= 40}
    if len(series) < groups * 3:
        raise ValueError("有效股票样本不足，请增大候选池规模")

    bench_series = None
    if bench_code:
        try:
            bench_series = await adapters.fetch_kline(bench_code, hist + n + 25)
        except Exception:
            bench_series = None

    date_maps = {code: {row["date"]: idx for idx, row in enumerate(kl)} for code, kl in series.items()}
    ref_code = max(series, key=lambda c: len(series[c]))
    ref_dates = [row["date"] for row in series[ref_code]]

    code_cache = {}
    for code, kl in series.items():
        arr = kline_to_arrays(kl)
        code_cache[code] = {
            "closes": arr["close"],
            "smap": {k: compute_factor_series(k, arr) for k in tech_keys},
            "kline": kl,
        }

    # 快照特征（最新值作静态特征，所有调仓日同值；含前视风险，仅探索用）
    row_by_code = {r["code"]: r for r in pool}
    snapshot_vals = {}
    if snap_keys:
        for code in series:
            r = row_by_code.get(code)
            if not r:
                continue
            sv = [snapshot_factor_value(r, k) for k in snap_keys]
            snapshot_vals[code] = sv if not any(v is None for v in sv) else None

    cost_rate = round_trip_cost_rate(commission_rate, stamp_duty, slippage) if apply_cost else 0.0

    ic_series = []
    bucket_returns = [[] for _ in range(groups)]
    long_short_points = []
    top_group_returns = []
    bench_by_date = {}
    cum = 1.0
    cum_top = 1.0

    bench_date_idx = {row["date"]: idx for idx, row in enumerate(bench_series)} if bench_series else {}

    for t in range(25, len(ref_dates) - n, max(1, n)):
        date_t = ref_dates[t]
        cross_feats, cross_codes, cross_rets = [], [], []
        for code in series:
            i = date_maps[code].get(date_t)
            if i is None or i < 20 or i + n >= len(code_cache[code]["kline"]):
                continue
            cc = code_cache[code]
            fvals = []
            ok = True
            for k in tech_keys:
                v = series_at(cc["smap"][k], i)
                if v is None:
                    ok = False
                    break
                fvals.append(v)
            if not ok or cc["closes"][i] == 0:
                continue
            # 快照特征（静态）
            if snap_keys:
                sv = snapshot_vals.get(code)
                if not sv:
                    continue
                fvals.extend(sv)
            # 涨跌停约束：t 日涨停封板买不进；t+n 日跌停卖不出
            closes = cc["closes"]
            limit = _price_limit_ratio(code, is_st.get(code, False))
            if i >= 1 and closes[i - 1] != 0 and closes[i] / closes[i - 1] - 1.0 >= limit - 1e-4:
                continue
            if closes[i + n - 1] != 0 and closes[i + n] / closes[i + n - 1] - 1.0 <= -limit + 1e-4:
                continue
            cross_codes.append(code)
            cross_feats.append(fvals)
            cross_rets.append(float(closes[i + n] / closes[i] - 1))
        if len(cross_codes) < groups * 2:
            continue

        Xp = _apply_preprocess(np.array(cross_feats, dtype=np.float64), params)
        preds = model.predict(Xp).tolist()

        ic = _pearson(preds, cross_rets)
        rank_ic = _spearman(preds, cross_rets)
        ic_series.append({"date": date_t, "ic": ic, "rankIc": rank_ic, "sample": len(cross_codes)})

        sorted_preds = sorted(preds)
        day_buckets = [[] for _ in range(groups)]
        for fv, fret in zip(preds, cross_rets):
            b = bucket_index(fv, sorted_preds, groups)
            day_buckets[b].append(fret)
            bucket_returns[b].append(fret)

        if day_buckets[0] and day_buckets[-1]:
            top_ret = mean(day_buckets[-1])
            bottom_ret = mean(day_buckets[0])
            ls_ret = top_ret - bottom_ret
            # 多空双腿各一份往返成本（旧版仅扣一份）
            net_ls = ls_ret - 2.0 * cost_rate
            net_top = top_ret - cost_rate
            cum *= (1.0 + net_ls)
            cum_top *= (1.0 + net_top)
            long_short_points.append({
                "date": date_t, "longShort": net_ls, "cum": cum - 1.0,
                "topCum": cum_top - 1.0, "gross": ls_ret,
            })
            top_group_returns.append(net_top)
            if bench_series and bench_date_idx:
                bi = bench_date_idx.get(date_t)
                if bi is not None and bi + n < len(bench_series):
                    bc = [row["close"] for row in bench_series]
                    bench_by_date[date_t] = bc[bi + n] / bc[bi] - 1

    if not ic_series:
        raise ValueError("有效截面样本不足，无法完成 ML 信号分层回测")

    group_summary = [
        {"group": idx + 1, "avgReturn": mean(rets) if rets else 0.0, "sample": len(rets)}
        for idx, rets in enumerate(bucket_returns)
    ]

    ls_returns = [p["longShort"] for p in long_short_points]
    ls_equity = [1.0]
    for r in ls_returns:
        ls_equity.append(ls_equity[-1] * (1.0 + r))

    # 调仓间隔 n 日 → 年化采样数 = 252/n（旧版误用 252，年化收益高估 n 倍、Sharpe 高估 √n 倍）
    ppy = 252.0 / max(1, n)

    metrics = {
        "cumulativeReturn": cum - 1.0,
        "annualizedReturn": annualized_return(ls_returns, ppy),
        "annualizedVolatility": annualized_volatility(ls_returns, ppy),
        "sharpe": sharpe_ratio(ls_returns, periods_per_year=ppy),
        "sortino": sortino_ratio(ls_returns, periods_per_year=ppy),
        "maxDrawdown": max_drawdown(ls_equity),
        "calmar": calmar_ratio(ls_returns, ppy),
        "winRate": win_rate(ls_returns),
        "topGroupCumReturn": cum_top - 1.0,
        "topGroupAnnualized": annualized_return(top_group_returns, ppy),
        "rebalanceCount": len(ic_series),
        "costRate": cost_rate,
        "applyCost": apply_cost,
    }

    bench_metrics = None
    if bench_by_date:
        # 按日期交集对齐策略与基准收益
        aligned = [(p["longShort"], bench_by_date[p["date"]])
                   for p in long_short_points if p["date"] in bench_by_date]
        aligned_ls = [a for a, _ in aligned]
        aligned_bench = [b for _, b in aligned]
        bench_cum = 1.0
        for r in aligned_bench:
            bench_cum *= (1.0 + r)
        ab = (_alpha_beta(aligned_bench, aligned_ls)
              if len(aligned_bench) > 1 else {"alpha": 0.0, "beta": 0.0})
        bench_metrics = {
            "code": bench_code,
            "cumulativeReturn": bench_cum - 1.0,
            "annualizedReturn": annualized_return(aligned_bench, ppy),
            "annualizedVolatility": annualized_volatility(aligned_bench, ppy),
            "sharpe": sharpe_ratio(aligned_bench, periods_per_year=ppy),
            "maxDrawdown": max_drawdown(_equity_curve(aligned_bench)),
            **ab,
        }

    ic_stats = information_coefficient_stats([p["ic"] for p in ic_series], ppy)
    mean_rank_ic = mean([p["rankIc"] for p in ic_series])

    return {
        "factorLabel": f"ML模型({mid})", "groups": groups, "n": n, "modelId": mid,
        "meanIc": ic_stats["meanIc"], "meanRankIc": mean_rank_ic,
        "icWinRate": ic_stats["icWinRate"], "icIr": ic_stats["icIr"],
        "icSeries": ic_series, "groupSummary": group_summary, "longShort": long_short_points,
        "metrics": metrics, "benchmark": bench_metrics,
        "universeSize": len(pool), "effectiveStocks": len(series), "rebalanceCount": len(ic_series),
        "survivorshipBiasWarning": "候选池为当前上市股票快照，已退市股票不在回测池中，历史收益可能系统性高估",
    }


def _equity_curve(returns: list[float]) -> list[float]:
    eq = [1.0]
    for r in returns:
        eq.append(eq[-1] * (1.0 + r))
    return eq


def _alpha_beta(bench: list[float], strat: list[float]) -> dict:
    n = len(bench)
    if n < 2:
        return {"alpha": 0.0, "beta": 0.0}
    mx, my = mean(bench), mean(strat)
    num = sum((bench[i] - mx) * (strat[i] - my) for i in range(n))
    den = sum((b - mx) ** 2 for b in bench)
    b = 0.0 if den == 0 else num / den
    return {"alpha": my - b * mx, "beta": b}
