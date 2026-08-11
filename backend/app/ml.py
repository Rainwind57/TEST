"""机器学习模块：GBDT 预测未来收益，时序交叉验证 + OOS 评估。

防泄漏要点：
- 目标 = 未来 N 日收益（closes[i+n]/closes[i]-1），仅用 t 及之前信息
- 时序 CV：Purged Walk-Forward，严禁随机 shuffle
- 训练/验证/测试三段指标，防止过拟合自夸
- 模型 joblib 落盘 + 元数据
"""
import os
import json
import ast
import re
import asyncio
import datetime
import logging
from collections import defaultdict

import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from . import adapters, db

logger = logging.getLogger(__name__)
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

# 内置技术因子的最长回看窗口：dist_52w_high/low 完整窗口 240 日（momentum120 次之，120 日）。
# ML 训练/评估默认 hist 必须覆盖它，否则「开箱即用」必报有效样本不足。
_MAX_FACTOR_LOOKBACK = 240
# ML 训练/评估/寻优的默认历史长度：1024 覆盖全部内置长周期因子（新浪封顶 1023 可及）
ML_DEFAULT_HIST = 1024


def min_hist_for_ml(n: int = 5) -> int:
    """ML 内置技术因子所需最小历史长度：最长因子窗口 + 持有期 + 1（样本起点）。"""
    return _MAX_FACTOR_LOOKBACK + n + 1


def _resolve_boards(board: str, boards: list[str] | None) -> list[str]:
    """板块解析：优先用 boards（多选），否则回退单 board（向后兼容旧前端）。"""
    bs = [b for b in (boards or []) if b]
    return bs or [board or "all"]


def _empty_pool_error(board: str, pool_size: int) -> ValueError:
    """候选池为空时的结构化错误：区分「上游行情源故障」与「板块无匹配股票」。"""
    health = adapters.market_list_health()
    if health.get("degraded"):
        return ValueError(
            f"候选池为空：上游行情列表获取失败（{health.get('last_error', '网络故障/被限流')}）。"
            f"这是数据源问题而非参数问题，请稍后重试，或先到行情页确认列表可加载。"
        )
    return ValueError(
        f"候选池为空：板块「{board}」当前无匹配股票（poolSize={pool_size}）。"
        f"请检查板块范围或候选池规模。"
    )


async def _kline_retry(kline_fn, code: str, days: int, attempts: int = 3,
                      delay: float = 0.5) -> list[dict]:
    """K线拉取带指数退避重试。空结果也重试（上游限流常返回空而非抛错）。"""
    last_error = None
    for attempt in range(attempts):
        try:
            kl = await kline_fn(code, days)
            if kl:
                return kl
            # 上游限流返回空也重试（旧版把空当作成功直接返回，单次限流就丢票）
            if attempt == attempts - 1:
                return []
        except Exception as e:
            last_error = e
            if "Expecting value" in str(e) or "JSON" in str(type(e).__name__):
                return []
        if attempt < attempts - 1:
            await asyncio.sleep(delay * (2 ** attempt))
    if last_error:
        logger.warning("K线拉取失败 code=%s days=%s error=%s", code, days, last_error)
    return []

# 需要额外拉取的快照因子（行情快照 row 不自带，须调财务/资金流接口）
_FINANCE_KEYS = {"roe", "net_margin", "revenue_yoy", "profit_yoy",
                 "gross_margin", "debt_ratio", "eps", "bps", "roa"}
_MONEYFLOW_KEYS = {"main_net_pct", "north_holding_pct"}
_FIN_TO_FIELD = {
    "roe": "roe", "net_margin": "netMargin", "revenue_yoy": "revenueYoY",
    "profit_yoy": "profitYoY", "gross_margin": "grossMargin", "debt_ratio": "debtRatio",
    "eps": "eps", "bps": "bps", "roa": "roa",
}


async def _enrich_pool_extra(pool: list[dict], snap_keys: list[str]) -> None:
    """对候选池逐股拉取财务/资金流字段，就地写入 row（与 selection.fetch_extra 对齐）。

    旧版 ML 只喂 pe/pb/turnover 三个行情自带字段；扩展后 ep/bp/mkt_cap/roe 等全可入模型，
    训练与推理（score_latest/backtest_model）共用同一份 feature_names 同构对齐。
    """
    need_fin = bool(set(snap_keys) & _FINANCE_KEYS)
    need_mf = bool(set(snap_keys) & _MONEYFLOW_KEYS)
    if not need_fin and not need_mf:
        return
    sem = asyncio.Semaphore(10)

    async def one(row):
        code = row["code"]
        async with sem:
            if need_fin:
                try:
                    fin = await adapters.fetch_finance_summary(code)
                except Exception:
                    fin = {}
                for k, f in _FIN_TO_FIELD.items():
                    row[k] = fin.get(f) if fin.get(f) is not None else 0.0
            if need_mf:
                if "main_net_pct" in snap_keys:
                    try:
                        mf = await adapters.fetch_money_flow(code)
                    except Exception:
                        mf = {}
                    row["main_net_pct"] = mf.get("mainNetPct")
                if "north_holding_pct" in snap_keys:
                    try:
                        nh = await adapters.fetch_north_holding(code)
                    except Exception:
                        nh = {}
                    row["north_holding_pct"] = nh.get("holdRatio")

    await asyncio.gather(*(one(r) for r in pool))


async def build_dataset(board: str = "all", pool_size: int = 100, n: int = 5,
                        hist: int = 240, progress_cb=None,
                        use_snapshot: bool = False,
                        asset_class: str = "a-share",
                        start_date: str | None = None,
                        end_date: str | None = None,
                        boards: list[str] | None = None,
                        selected_factors: list[str] | None = None) -> dict:
    """构建 ML 数据集：候选池每只股票算全部量价因子 + 未来 N 日收益。

    返回 {features, target, codes, dates, feature_names}。
    use_snapshot=True 时追加全部快照因子（含财务/资金流，需额外拉取）作为静态特征
    （含前视风险，仅探索用；推理 score_latest/backtest_model 会按 feature_names 一致拉取）。
    selected_factors 不为空时仅使用指定因子子集（技术因子 key 或快照因子 key），
    未传则默认使用全部因子。
    asset_class=future 时候选池取期货主力连续合约（FUTURE_UNIVERSE），K 线走期货适配器，
    快照因子不适用（期货无财务/资金流字段），强制忽略。
    """
    if asset_class == "future":
        pool = [{"code": c, "name": c} for c in adapters.FUTURE_UNIVERSE[:pool_size]]
        kline_fn = adapters.fetch_future_kline
        snapshot_keys = []
    else:
        # 多板块 OR 合并取池（板块可组合；hs300/zz500 在 adapters 内走真实成分股）
        pool = await adapters.fetch_market_list_multi(_resolve_boards(board, boards), pool_size)
        kline_fn = adapters.fetch_kline
        # sector 为类别因子（direction=0），不可作数值特征，否则 np.float64 转型崩溃
        snapshot_keys = [k for k in SNAPSHOT_FACTORS.keys()
                         if use_snapshot and SNAPSHOT_FACTORS[k].get("format") != "categorical"]
    # 候选池为空：报「上游行情源故障」而非误导性的「样本不足」（P0 可观测性）
    if not pool:
        raise _empty_pool_error(board, pool_size)
    # 历史长度钳制：内置长周期因子（如 momentum120/dist_52w_high 240 日窗口）需要足够历史，
    # 用户传过小的 hist 时自动抬升，避免默认参数下「开箱即失败」
    hist = max(int(hist), min_hist_for_ml(n))
    codes = [row["code"] for row in pool]
    sem_factor_keys = [k for k in FACTORS]
    # 因子选择：selected_factors 非空时只保留用户勾选的因子
    if selected_factors:
        sel = set(selected_factors)
        sem_factor_keys = [k for k in sem_factor_keys if k in sel]
        snapshot_keys = [k for k in snapshot_keys if k in sel]
    if snapshot_keys:
        await _enrich_pool_extra(pool, snapshot_keys)
    row_by_code = {r["code"]: r for r in pool}
    total = len(codes)

    rows = []
    labels = []
    meta_codes = []
    meta_dates = []
    stat = {"total": total, "kline_ok": 0, "kline_fail": 0, "kline_short": 0, "factor_missing": 0}
    for idx, code in enumerate(codes):
        if progress_cb:
            progress_cb(idx + 1, total)
        try:
            kline = await kline_fn(code, hist)
        except Exception as e:
            stat["kline_fail"] += 1
            logger.warning("ML数据集拉取K线失败 code=%s: %s", code, e)
            continue
        if len(kline) < 60 + n:
            stat["kline_short"] += 1
            continue
        stat["kline_ok"] += 1
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
                stat["factor_missing"] += 1
                continue
            # 快照特征（最新值作静态特征，含前视风险，仅探索用）
            if use_snapshot:
                r = row_by_code.get(code)
                if not r:
                    continue
                sv = [snapshot_factor_value(r, k) for k in snapshot_keys]
                if any(v is None for v in sv):
                    stat["factor_missing"] += 1
                    continue
                fvals.extend(sv)
            fret = closes[i + n] / closes[i] - 1.0
            rows.append(fvals)
            labels.append(fret)
            meta_codes.append(code)
            meta_dates.append(arr["date"][i])

    if not rows:
        raise ValueError(
            f"有效样本不足，无法构建数据集：候选池 {stat['total']} 只，K线拉取失败 {stat['kline_fail']} 只，"
            f"历史不足 {60 + n} 日被剔除 {stat['kline_short']} 只，因子缺失切片 {stat['factor_missing']} 个。"
            f"请增大候选池规模(poolSize)、加长历史(hist)或放宽板块范围后再试。"
        )

    # 按 (date, code) 排序后再切分，杜绝时序 CV 训练集混入晚于测试集的日期样本。
    order = sorted(range(len(rows)), key=lambda k: (meta_dates[k], meta_codes[k]))
    rows = [rows[k] for k in order]
    labels = [labels[k] for k in order]
    meta_codes = [meta_codes[k] for k in order]
    meta_dates = [meta_dates[k] for k in order]

    # 训练/验证时间段过滤（分时段训练：只保留落在 [start_date, end_date] 内的样本）
    if start_date or end_date:
        keep = [k for k, d in enumerate(meta_dates)
                if (not start_date or d >= start_date) and (not end_date or d <= end_date)]
        if not keep:
            raise ValueError(
                f"指定时间段（{start_date or '不限'} ~ {end_date or '不限'}）内有效样本为 0。"
                f"请调整时间段范围，或增大候选池/加长历史(hist)后再试。"
            )
        rows = [rows[k] for k in keep]
        labels = [labels[k] for k in keep]
        meta_codes = [meta_codes[k] for k in keep]
        meta_dates = [meta_dates[k] for k in keep]

    # 快照特征前视偏差截断：仅保留最近 60 个交易日样本（与回测侧一致）
    snapshot_warning = None
    if use_snapshot and snapshot_keys:
        from datetime import date as _date
        today_str = _date.today().isoformat()
        past_dates = sorted(set(d for d in meta_dates if d <= today_str))
        cutoff = past_dates[-60] if len(past_dates) >= 60 else (past_dates[0] if past_dates else None)
        if cutoff:
            keep = [k for k, d in enumerate(meta_dates) if d >= cutoff]
            dropped = len(rows) - len(keep)
            if dropped > 0:
                rows = [rows[k] for k in keep]
                labels = [labels[k] for k in keep]
                meta_codes = [meta_codes[k] for k in keep]
                meta_dates = [meta_dates[k] for k in keep]
                snapshot_warning = (
                    f"快照因子（财务/资金流）仅最近 60 交易日有效，"
                    f"已截断 {dropped} 条更早样本（截止日 {cutoff}），"
                    f"避免历史截面使用当前财报数据导致前视偏差"
                )

    return {
        "features": np.array(rows, dtype=np.float64),
        "target": np.array(labels, dtype=np.float64),
        "codes": meta_codes,
        "dates": meta_dates,
        "feature_names": sem_factor_keys + snapshot_keys,
        "n": n,
        "snapshotWarning": snapshot_warning,
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




def purged_walk_forward_split_by_dates(dates: list[str], n_splits: int = 5,
                                      test_ratio: float = 0.2, gap_days: int = 5):
    """时序 Walk-Forward 分割（按交易日 gap，杜绝同日 train/test 重叠）。

    dates 已按时间升序排列。去重后按日期组切块，训练区末尾与测试区起始之间至少间隔
    gap_days 个交易日（而非样本序号），确保整日不会被 train/test 拆分。
    返回 [(train_idx, test_idx), ...]，idx 为原样本序号。
    """
    unique_dates = sorted(set(dates))
    # 日期 → 该日期样本在 dates 中的所有下标
    date_to_indices = {d: [] for d in unique_dates}
    for idx, d in enumerate(dates):
        date_to_indices[d].append(idx)

    n_dates = len(unique_dates)
    fold_dates = n_dates // n_splits

    splits = []
    for k in range(n_splits - 1):
        train_end_date_idx = (k + 1) * fold_dates
        test_start_date_idx = train_end_date_idx + gap_days
        test_end_date_idx = min((k + 2) * fold_dates, n_dates)
        if test_start_date_idx >= n_dates:
            break

        train_dates = unique_dates[:train_end_date_idx]
        test_dates = unique_dates[test_start_date_idx:test_end_date_idx]

        train_idx = np.array([i for d in train_dates for i in date_to_indices[d]], dtype=int)
        test_idx = np.array([i for d in test_dates for i in date_to_indices[d]], dtype=int)

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        splits.append((train_idx, test_idx))

    return splits


def _find_cv_splits_by_dates(dates: list[str], n_splits: int, gap_days: int):
    """按日期 purged CV 自动降级。"""
    for ns in (n_splits, *range(max(2, n_splits - 1), 1, -1)):
        splits = purged_walk_forward_split_by_dates(dates, n_splits=ns, gap_days=gap_days)
        if any(len(tr) >= 50 and len(te) >= 10 for tr, te in splits):
            if ns != n_splits:
                logger.info("CV 折数从 %s 自动降级为 %s（按交易日 gap=%s）", n_splits, ns, gap_days)
            return splits
    return []


def evaluate_dataset(dataset: dict, model_type: str = "gbdt", n_splits: int = 5,
                     gap: int = 5, progress_cb=None) -> dict:
    """时序 CV 评估：每折在训练折内 fit 预处理（缩尾+标准化）再 apply 到测试折，
    杜绝预处理泄漏；OOS Sharpe 按调仓日聚合多空序列计算（旧版单元素列表恒为 0）。
    gap 按交易日计数（非样本序号），杜绝同日 train/test 重叠。"""
    X_raw = dataset["features"]
    y = dataset["target"]
    dates = dataset["dates"]
    feature_names = dataset["feature_names"]
    n_hold = dataset.get("n", gap)
    gap_days = max(gap, n_hold)
    n = len(y)

    splits = _find_cv_splits_by_dates(dates, n_splits, gap_days)
    if not splits:
        raise ValueError(
            f"样本不足以做时序交叉验证：当前样本 {n} 条，分不出（训练≥50 & 测试≥10）的有效折。"
            f"建议增大候选池/历史长度，或将 CV 折数调小（当前 {n_splits}）、缩小 gap（当前 {gap}）。"
        )

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
        raise ValueError(
            f"样本不足以完成任何一折训练：总样本 {n} 条、折数 {len(splits)}，每折训练<50 或测试<10。"
            f"建议增大候选池/历史长度，或减小 CV 折数/gap（当前 n_splits={n_splits}, gap={gap}）。"
        )

    valid_mask = ~np.isnan(all_pred)
    valid_pred = all_pred[valid_mask]
    valid_y = y[valid_mask]
    valid_dates = [dates[i] for i in range(n) if valid_mask[i]]
    # 按日算 IC 再取均值（避免跨日期混池导致虚假 IC）
    date_ic = defaultdict(list)
    for pred, actual, d in zip(valid_pred, valid_y, valid_dates):
        date_ic[d].append((pred, actual))
    per_date_ic = []
    for d in sorted(date_ic):
        items = date_ic[d]
        if len(items) >= 3:
            ps = np.array([p for p, _ in items])
            ys = np.array([y for _, y in items])
            per_date_ic.append(_pearson(ps, ys))
    overall_ic = float(np.mean(per_date_ic)) if per_date_ic else 0.0
    overall_rank_ic = 0.0  # rank_ic 同理按日算；保持兼容不重算
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


def train_final_model(dataset: dict, model_type: str = "gbdt", params: dict | None = None) -> dict:
    """用全量数据训练最终模型并落盘，返回模型元数据。params 非空时使用寻优超参。

    预处理参数（缩尾分位数 + 均值 + 标准差）随 joblib 一起落盘 + sidecar JSON 元数据，
    推理时复现预处理（旧版只存 model+feature_names，推理无法复现缩尾/标准化）。
    """
    X_raw = dataset["features"]
    y = dataset["target"]
    feature_names = dataset["feature_names"]

    pparams = _fit_preprocess(X_raw)
    X = _apply_preprocess(X_raw, pparams)
    model = _build_model(model_type, params)
    model.fit(X, y)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mid = f"mlmodel_{ts}"
    path = os.path.join(ML_DIR, f"{mid}.joblib")
    joblib.dump({
        "model": model,
        "feature_names": feature_names,
        "model_type": model_type,
        "preprocess": pparams,
    }, path)

    meta = {
        "id": mid, "path": path, "modelType": model_type,
        "featureNames": feature_names, "nSamples": len(y),
        "trainedAt": datetime.datetime.now().isoformat(),
    }
    # 特征重要性随训练写入侧车 JSON：调参接口读 meta 无需再反序列化整包模型
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        meta["featureImportance"] = sorted(
            [{"feature": feature_names[i], "importance": float(imp[i])}
             for i in range(min(len(feature_names), len(imp)))],
            key=lambda x: x["importance"], reverse=True)
    try:
        with open(os.path.join(ML_DIR, f"{mid}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return meta


def optimize_model(dataset: dict, model_type: str = "lightgbm",
                   n_splits: int = 5, gap: int = 5,
                   n_trials: int = 30, progress_cb=None,
                   cancel_event=None) -> dict:
    """Optuna 对 ML 模型超参寻优（旧版 Optuna 仅接因子回测，ML 调参未闭环）。

    目标 = 时序 Walk-Forward 最后一折 OOS Sharpe（按调仓日聚合多空），
    杜绝全样本调参后宣称高收益。返回 {best_params, oos_metrics, is_metrics, trials}，
    并把超参+指标写 ml_models/<ts>_opt.json 作实验追踪。
    """
    import optuna
    X_raw = dataset["features"]
    y = dataset["target"]
    dates = dataset["dates"]
    n = len(y)
    splits = _find_cv_splits_by_dates(dates, n_splits, gap)
    if not splits:
        raise ValueError("样本不足以做时序交叉验证")

    # 取倒数第2折作为 valid（调参用），最后一折作 holdout（独立 OOS 报告）
    if len(splits) >= 2:
        valid_train_idx, valid_test_idx = splits[-2]
        holdout_train_idx, holdout_test_idx = splits[-1]
    else:
        valid_train_idx, valid_test_idx = splits[-1]
        holdout_train_idx, holdout_test_idx = splits[-1]

    if len(valid_train_idx) < 50 or len(valid_test_idx) < 10:
        raise ValueError("验证折样本不足，需更多数据或减小 gap/n_splits")

    def _eval_oos_sharpe(params: dict) -> float:
        """valid 折 IS 训练 → OOS 评估，返回 OOS Sharpe。"""
        try:
            pp = _fit_preprocess(X_raw[valid_train_idx])
            Xtr = _apply_preprocess(X_raw[valid_train_idx], pp)
            Xte = _apply_preprocess(X_raw[valid_test_idx], pp)
            ytr, yte = y[valid_train_idx], y[valid_test_idx]
            m = _build_model(model_type, params)
            m.fit(Xtr, ytr)
            pred = m.predict(Xte)
            oos_records = [(dates[valid_test_idx[j]], float(pred[j]), float(yte[j]))
                           for j in range(len(valid_test_idx))]
            return _oos_sharpe_by_date(oos_records)
        except Exception:
            return -1e9

    def _objective(trial: optuna.Trial) -> float:
        if model_type == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", -1, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127, step=8),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
        else:  # gbdt
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }
        return _eval_oos_sharpe(params)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    for i in range(n_trials):
        if cancel_event and cancel_event.is_set():
            logger.info("ML 寻优被用户取消（已完成 %d/%d trials）", i, n_trials)
            break
        if progress_cb:
            progress_cb(i + 1, n_trials)
        study.optimize(_objective, n_trials=1, catch=(Exception,))

    best_params = study.best_params if study.best_trial else {}
    # 用最优参数在 valid 折算 IS 指标
    pp = _fit_preprocess(X_raw[valid_train_idx])
    Xtr = _apply_preprocess(X_raw[valid_train_idx], pp)
    Xte = _apply_preprocess(X_raw[valid_test_idx], pp)
    ytr, yte = y[valid_train_idx], y[valid_test_idx]
    best_model = _build_model(model_type, best_params)
    best_model.fit(Xtr, ytr)
    pred_is = best_model.predict(Xtr)
    pred_oos = best_model.predict(Xte)
    is_sharpe = _oos_sharpe_by_date(
        [(dates[valid_train_idx[j]], float(pred_is[j]), float(ytr[j])) for j in range(len(valid_train_idx))])
    oos_sharpe = _oos_sharpe_by_date(
        [(dates[valid_test_idx[j]], float(pred_oos[j]), float(yte[j])) for j in range(len(valid_test_idx))])
    oos_ic = _pearson(pred_oos, yte)
    oos_rank_ic = _spearman(pred_oos, yte)

    # 独立 holdout 折评估（仅在≥2折时才有独立 holdout）
    holdout_sharpe = None
    if len(splits) >= 2 and len(holdout_test_idx) >= 10:
        pp_h = _fit_preprocess(X_raw[holdout_train_idx])
        Xtr_h = _apply_preprocess(X_raw[holdout_train_idx], pp_h)
        Xte_h = _apply_preprocess(X_raw[holdout_test_idx], pp_h)
        ytr_h, yte_h = y[holdout_train_idx], y[holdout_test_idx]
        best_model_h = _build_model(model_type, best_params)
        best_model_h.fit(Xtr_h, ytr_h)
        pred_h = best_model_h.predict(Xte_h)
        holdout_sharpe = _oos_sharpe_by_date(
            [(dates[holdout_test_idx[j]], float(pred_h[j]), float(yte_h[j]))
             for j in range(len(holdout_test_idx))])
        holdout_ic = _pearson(pred_h, yte_h)

    trials = [
        {"number": t.number, "value": t.value, "params": t.params}
        for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    result = {
        "modelType": model_type,
        "bestParams": best_params,
        "validSharpe": is_sharpe,
        "validOosSharpe": oos_sharpe,
        "validOosIc": oos_ic,
        "validOosRankIc": oos_rank_ic,
        "holdoutSharpe": holdout_sharpe,
        "holdoutIc": holdout_ic if holdout_sharpe is not None else None,
        "holdoutDate": dates[holdout_test_idx[0]] if holdout_sharpe is not None and len(holdout_test_idx) else None,
        "splitDate": dates[valid_test_idx[0]] if len(valid_test_idx) else None,
        "nTrials": len(trials),
        "trials": trials,
        "holdoutAvailable": holdout_sharpe is not None,
    }

    # 用最优参数全量训练并落盘最终模型
    if best_params:
        try:
            final_meta = train_final_model(dataset, model_type, best_params)
            result["finalModel"] = final_meta
        except Exception as e:
            logger.warning("寻优后全量训练失败: %s", e)

    # 实验追踪：超参 + 指标落盘（旧版模型仅文件名时间戳，无法回溯参数）
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        with open(os.path.join(ML_DIR, f"opt_{ts}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return result


def list_models() -> list[dict]:
    out = []
    if not os.path.isdir(ML_DIR):
        return out
    from .numpy_factors import _FACTOR_SERIES_FN as _KNOWN_FACTORS
    from .factors import SNAPSHOT_FACTORS as _SNAPSHOT
    _valid_set = set(_KNOWN_FACTORS) | (set(_SNAPSHOT) - {"sector"})
    for f in sorted(os.listdir(ML_DIR), reverse=True):
        if f.endswith(".joblib"):
            mid = f.replace(".joblib", "")
            entry = {"id": mid, "file": f}
            try:
                meta = load_model_meta(mid)
                if meta:
                    entry["modelType"] = meta.get("modelType", "gbdt")
                    fns = meta.get("featureNames", [])
                    entry["featureNames"] = fns
                    entry["nFeatures"] = len(fns)
                    unknown = [k for k in fns if k not in _valid_set]
                    entry["computable"] = len(unknown) == 0
                    if unknown:
                        entry["unknownFeatures"] = unknown[:8]
            except Exception:
                pass
            out.append(entry)
    return out


def delete_model(mid: str) -> bool:
    path = os.path.join(ML_DIR, f"{mid}.joblib")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def load_model_meta(mid: str) -> dict | None:
    """读取模型元数据（特征名、重要性、超参），供前端可视化与人工调参。

    优先读 sidecar JSON（训练/导入/手动创建时已写入 featureImportance），
    避免每次「调参」都 joblib.load 整包模型：重模型反序列化对 lightgbm 版本敏感，
    跨版本反序列化可能段错误崩溃 worker，前端表现为 network error（P0）。
    仅对无重要性的旧模型才回退加载 bundle，且失败时返回 JSON 已有信息而非崩溃。
    """
    meta_path = os.path.join(ML_DIR, f"{mid}.json")
    meta = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = None
    path = os.path.join(ML_DIR, f"{mid}.joblib")
    if not os.path.exists(path):
        return None

    # JSON 已含特征名 + 重要性：直接返回，不反序列化模型包（调参热路径）
    if meta and meta.get("featureNames") and meta.get("featureImportance") is not None:
        result = {
            "id": mid,
            "modelType": meta.get("modelType", "gbdt"),
            "featureNames": meta["featureNames"],
            "featureImportance": meta["featureImportance"],
        }
        for k in ("trainedAt", "nSamples", "name", "manual", "rule",
                  "threshold", "imported", "sourceFile", "featureWeights"):
            if k in meta:
                result[k] = meta[k]
        return result

    # 旧模型（JSON 缺重要性）：回退加载 bundle 提取；失败返回 JSON 已有信息（不崩溃）
    feature_names = (meta or {}).get("featureNames") or []
    model_type = (meta or {}).get("modelType", "gbdt")
    importances = []
    try:
        bundle = joblib.load(path)
        feature_names = bundle.get("feature_names") or feature_names
        model_type = bundle.get("model_type") or model_type
        model = bundle.get("model")
        if model is not None and hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            importances = [{"feature": feature_names[i], "importance": float(imp[i])}
                           for i in range(min(len(feature_names), len(imp)))]
    except Exception:
        importances = []
    result = {
        "id": mid, "modelType": model_type, "featureNames": feature_names,
        "featureImportance": sorted(importances, key=lambda x: x["importance"], reverse=True),
    }
    if meta:
        for k in ("trainedAt", "nSamples", "name", "manual", "rule",
                  "threshold", "imported", "sourceFile", "featureWeights"):
            if k in meta:
                result[k] = meta[k]
    return result


def _build_model(model_type: str, params: dict | None = None):
    """构建模型。旧版仅 sklearn GBDT 且超参硬编码（200/4/0.05）；
    现新增 lightgbm（已在 requirements 声明却从未接线）并支持外部传超参。

    params 非空时用于 Optuna 调参寻优；为空时用合理默认值。
    """
    p = params or {}
    if model_type == "gbdt":
        return GradientBoostingRegressor(
            n_estimators=int(p.get("n_estimators", 200)),
            max_depth=int(p.get("max_depth", 4)),
            learning_rate=float(p.get("learning_rate", 0.05)),
            subsample=float(p.get("subsample", 0.8)),
            random_state=42,
        )
    if model_type == "lightgbm":
        import lightgbm as lgb  # 已在 requirements 声明，首次调用才 import
        return lgb.LGBMRegressor(
            n_estimators=int(p.get("n_estimators", 300)),
            max_depth=int(p.get("max_depth", -1)),
            learning_rate=float(p.get("learning_rate", 0.05)),
            num_leaves=int(p.get("num_leaves", 31)),
            subsample=float(p.get("subsample", 0.8)),
            colsample_bytree=float(p.get("colsample_bytree", 0.8)),
            reg_alpha=float(p.get("reg_alpha", 0.0)),
            reg_lambda=float(p.get("reg_lambda", 0.0)),
            random_state=42,
            verbose=-1,
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


def _apply_feature_weights(Xp: np.ndarray, feature_names: list[str],
                           feature_weights: dict) -> np.ndarray:
    """人工调参：按特征名对输入做权重缩放（GBDT 黑盒下最直观的干预手段）。

    特征权重改变输入分布 → 决策树分裂路径变化 → 预测分排序变化。
    """
    if not feature_weights:
        return Xp
    Xw = Xp.copy()
    for i, name in enumerate(feature_names):
        w = feature_weights.get(name)
        if w is not None:
            Xw[:, i] = Xw[:, i] * float(w)
    return Xw


def _apply_threshold(preds: list[float], threshold: float | None) -> list[float]:
    """人工调参：预测分整体偏移（单调变换，不影响排序；用于选股绝对分过滤）。"""
    if not threshold:
        return preds
    return [p + float(threshold) for p in preds]


async def score_latest(mid: str, board: str = "all", pool_size: int = 100,
                       progress_cb=None, adjust: dict | None = None,
                       asset_class: str = "a-share",
                       boards: list[str] | None = None) -> list[dict]:
    """加载落盘模型对候选池最新截面打分，返回按预测分降序的打分列表。"""
    bundle = _load_model(mid)
    feature_names = bundle["feature_names"]
    model = bundle["model"]
    params = bundle["preprocess"]
    # sector 为类别因子不可作数值特征：从快照集合剔除，避免历史模型含 sector 时特征长度不匹配
    snap_set = set(SNAPSHOT_FACTORS) - {"sector"}

    # 校验技术因子名可计算（同 backtest_model）
    from .numpy_factors import _FACTOR_SERIES_FN as _KNOWN_FACTORS
    tech_keys = [k for k in feature_names if k not in snap_set]
    unknown_tech = [k for k in tech_keys if k not in _KNOWN_FACTORS]
    if unknown_tech:
        raise ValueError(
            f"模型中包含 {len(unknown_tech)} 个无法从K线实时计算的特征: {unknown_tech[:8]}..."
            f"（总数 {len(tech_keys)} 个技术因子）。"
            f"请使用包含真实因子名的模型，或重新训练时选择具体因子。"
        )

    if asset_class == "future":
        pool = [{"code": c, "name": c} for c in adapters.FUTURE_UNIVERSE[:pool_size]]
        kline_fn = adapters.fetch_future_kline
        snap_keys = []
    else:
        pool = await adapters.fetch_market_list_multi(_resolve_boards(board, boards), pool_size)
        kline_fn = adapters.fetch_kline
        snap_keys = [k for k in feature_names if k in snap_set]
    if not pool:
        raise _empty_pool_error(board, pool_size)
    if snap_keys:
        await _enrich_pool_extra(pool, snap_keys)
    sem = asyncio.Semaphore(min(50, max(15, len(pool))))
    collected = []

    async def one(row):
        code = row["code"]
        async with sem:
            kline = await _kline_retry(kline_fn, code, 260)
            if not kline:
                logger.warning("ML打分拉取K线失败 code=%s", code)
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
    if adjust:
        Xp = _apply_feature_weights(Xp, feature_names, adjust.get("featureWeights") or {})
    preds = model.predict(Xp)
    if adjust:
        preds = _apply_threshold(preds.tolist(), adjust.get("threshold"))
    else:
        preds = preds.tolist()
    # 人造模型规则过滤：rule 不为空时在打分后过滤不符合规则的样本
    model_rule = getattr(model, 'rule', '')
    if model_rule:
        keep_idx = _apply_rule_filter(collected, X, feature_names, model_rule)
        if len(keep_idx) < len(collected):
            collected = [collected[i] for i in keep_idx]
            preds = [preds[i] for i in keep_idx]
    rows = [{"code": r["code"], "name": r["name"], "score": float(p)}
            for r, p in zip(collected, preds)]
    rows.sort(key=lambda x: x["score"], reverse=True)
    for rank, r in enumerate(rows, start=1):
        r["rank"] = rank
    return rows


async def backtest_model(mid: str, board: str = "all", pool_size: int = 150, groups: int = 3,
                         n: int = 3, hist: int = 1024, commission_rate: float = 0.00025,
                         stamp_duty: float = 0.001, slippage: float = 0.001,
                         benchmark: str = "none", apply_cost: bool = True,
                         progress_cb=None, adjust: dict | None = None,
                         asset_class: str = "a-share",
                         start_date: str | None = None,
                         end_date: str | None = None,
                         config: dict | None = None,
                         boards: list[str] | None = None) -> dict:
    """用落盘模型预测分作为截面信号做分层回测，响应结构与 /api/select/backtest 一致，
    前端图表零成本复用。每个调仓日对每只股票取当日因子特征→模型预测分→分层。"""
    bundle = _load_model(mid)
    feature_names = bundle["feature_names"]
    model = bundle["model"]
    params = bundle["preprocess"]
    snap_set = set(SNAPSHOT_FACTORS) - {"sector"}
    tech_keys = [k for k in feature_names if k not in snap_set]

    # 校验技术因子名可计算：泛化特征（f0/f1...）无法用 compute_factor_series 从 K 线实时计算，
    # 早报错比拉完 150 只 K 线后逐日发现全量 tech_none 强得多
    from .numpy_factors import _FACTOR_SERIES_FN as _KNOWN_FACTORS
    unknown_tech = [k for k in tech_keys if k not in _KNOWN_FACTORS]
    if unknown_tech:
        raise ValueError(
            f"模型中包含 {len(unknown_tech)} 个无法从K线实时计算的特征: {unknown_tech[:8]}..."
            f"（总数 {len(tech_keys)} 个技术因子）。"
            f"这类泛化特征（如 f0/f1）缺少与量价因子的映射关系，"
            f"无法在回测中为历史截面对应日期计算因子值。"
            f"请使用包含真实因子名（如 momentum/rsi/volatility）的模型，"
            f"或重新训练时选择具体因子。"
        )

    groups = max(2, min(10, groups))
    n = max(1, n)
    # 历史钳制：内置长周期因子需要足够窗口（momentum120 / dist_52w_high 240 日）；
    # 上限 1500 日（约 6 年）：超过此值 Sina API 返回 null，Tencent 降级返回格式异常
    hist = max(int(hist), min_hist_for_ml(n))
    hist = min(hist, 1500)

    bench_code = BENCHMARKS.get(benchmark)
    if benchmark != "none" and bench_code is None:
        raise ValueError(f"未知基准: {benchmark}")

    if asset_class == "future":
        pool = [{"code": c, "name": c} for c in adapters.FUTURE_UNIVERSE[:pool_size]]
        kline_fn = adapters.fetch_future_kline
        snap_keys = []
    else:
        pool = await adapters.fetch_market_list_multi(_resolve_boards(board, boards), pool_size)
        kline_fn = adapters.fetch_kline
        # 推理侧与训练侧同构：feature_names 含财务/资金流字段时需拉取填充 row
        snap_keys = [k for k in feature_names if k in snap_set]
    # 候选池为空：报「上游行情源故障」而非误导性的「样本不足」
    if not pool:
        raise _empty_pool_error(board, pool_size)
    # 推理侧与训练侧同构：feature_names 含财务/资金流字段时需拉取填充 row
    if snap_keys:
        await _enrich_pool_extra(pool, snap_keys)
    codes = [row["code"] for row in pool]
    is_st = {} if asset_class == "future" else {
        r["code"]: ("ST" in r.get("name", "") or "*ST" in r.get("name", ""))
        for r in pool
    }
    # 并发上限收紧 + 单票重试：上游限流时 50 并发极易被掐，10~12 并发 + 3 次重试稳得多
    sem = asyncio.Semaphore(min(4, max(3, len(codes))))
    fetch_fail = 0

    async def fetch_one(code):
        nonlocal fetch_fail
        async with sem:
            kline = await _kline_retry(kline_fn, code, hist + n + 25)
            if not kline:
                fetch_fail += 1
                logger.warning("ML回测拉取K线失败 code=%s", code)
            return code, kline

    fetched = await asyncio.gather(*(fetch_one(c) for c in codes))
    series = {code: kl for code, kl in fetched if len(kl) >= 40}
    if len(series) < groups * 3:
        raise ValueError(
            f"有效股票样本不足：候选池 {len(pool)} 只，K线拉取失败 {fetch_fail} 只，"
            f"历史不足被剔除 {len(pool) - len(series) - fetch_fail} 只，K线≥40日的仅 {len(series)} 只（需 ≥{groups * 3}）。"
            f"请增大候选池规模或放宽板块范围。"
        )

    bench_series = None
    if bench_code:
        try:
            bench_series = await adapters.fetch_kline(bench_code, hist + n + 25)
        except Exception:
            bench_series = None

    # 快照特征：仅对最近 60 个交易日沿用（避免历史截面用今日财报数据→前视）
    ref_code = max(series, key=lambda c: len(series[c]))
    ref_dates = [row["date"] for row in series[ref_code]]
    snapshot_cutoff_date = None
    if snap_keys:
        from datetime import date as _date
        today_str = _date.today().isoformat()
        snapshot_cutoff_date = today_str
        # 向前数 60 个交易日得到允许区间
        past_dates = [d for d in ref_dates if d <= today_str]
        if len(past_dates) >= 60:
            snapshot_cutoff_date = past_dates[-60]
        # 含快照因子的模型：若未显式指定 start_date，自动从快照生效日起步，
        # 避免 96%+ 调仓日因前视跳过无意义遍历（大幅减少耗时且不丢有效截面）
        if not start_date and snapshot_cutoff_date:
            start_date = snapshot_cutoff_date

    code_cache = {}
    for code, kl in series.items():
        arr = kline_to_arrays(kl)
        code_cache[code] = {
            "closes": arr["close"],
            "opens": arr["open"] if "open" in arr else arr["close"],
            "volumes": arr["volume"],
            "smap": {k: compute_factor_series(k, arr) for k in tech_keys},
            "kline": kl,
        }

    # 快照特征（仅回测区间末端 60 日内有效，早期无法取当日值）
    row_by_code = {r["code"]: r for r in pool}
    snapshot_vals = {}
    if snap_keys:
        for code in series:
            r = row_by_code.get(code)
            if not r:
                continue
            sv = [snapshot_factor_value(r, k) for k in snap_keys]
            # 缺失快照值用 0.0 填充：财务 API 偶发失败不应导致整只股票被丢弃
            sv_filled = [v if v is not None else 0.0 for v in sv]
            snapshot_vals[code] = sv_filled

    cost_rate = round_trip_cost_rate(commission_rate, stamp_duty, slippage) if apply_cost else 0.0

    ic_series = []
    bucket_returns = [[] for _ in range(groups)]
    long_short_points = []
    top_group_returns = []
    bench_by_date = {}
    cum = 1.0
    cum_top = 1.0

    bench_date_idx = {row["date"]: idx for idx, row in enumerate(bench_series)} if bench_series else {}

    date_maps = {code: {row["date"]: i for i, row in enumerate(kl)} for code, kl in series.items()}

    # 长历史自适应步长：最多 200 个调仓日，n<10 时调仓密度足够；过密则自动扩大步长
    total_dates = len(ref_dates) - 60 - n
    step = max(n, total_dates // max(1, min(200, total_dates // 3)))
    step = max(1, min(step, 60))  # 步长上限 60 交易日
    if progress_cb:
        progress_cb(0, (total_dates // step) + 1)
    rebalance_idx = 0
    for t in range(60, len(ref_dates) - n, step):
        date_t = ref_dates[t]
        # 验证区间过滤（分时段验证：调仓日只落在 [start_date, end_date] 内）
        if start_date and date_t < start_date:
            continue
        if end_date and date_t > end_date:
            continue
        cross_feats, cross_codes, cross_rets, ic_rets = [], [], [], []
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
            # 快照特征：仅在允许日期区间内生效
            if snap_keys:
                if snapshot_cutoff_date and date_t < snapshot_cutoff_date:
                    continue  # 快照因子为当前值，回溯到历史截面即前视，跳过
                sv = snapshot_vals.get(code)
                if not sv:
                    # 缺失时用 0.0 填充（与 enrich 侧对齐，财务 API 偶发故障不丢票）
                    sv = [0.0] * len(snap_keys)
                fvals.extend(sv)
            # 涨跌停约束：t 日涨停封板买不进；t+n 日跌停卖不出（期货无涨跌停，跳过）
            closes = cc["closes"]
            if asset_class == "future":
                limit = None
            else:
                limit = _price_limit_ratio(code, is_st.get(code, False))
            # 涨跌停约束：t 日涨停封板买不进，剔除该样本；t+n 跌停/停牌不再剔除（保留真实收益分布）
            if limit is not None:
                if i >= 1 and closes[i - 1] != 0 and closes[i] / closes[i - 1] - 1.0 >= limit - 1e-4:
                    continue
            # T+1 入场：IC 用信号日收盘→未来收盘（衡量预测力）；P&L 用次日开盘→未来收盘（可实盘）
            ic_ret = float(closes[i + n] / closes[i] - 1)
            opens = cc.get("opens", closes)
            t1_entry = float(opens[i + 1]) if i + 1 < len(opens) else float(closes[i])
            t1_ret = float(closes[i + n] / t1_entry - 1) if t1_entry > 0 else ic_ret
            cross_codes.append(code)
            cross_feats.append(fvals)
            cross_rets.append(t1_ret)
            ic_rets.append(ic_ret)
        if len(cross_codes) < groups * 2:
            continue

        Xp = _apply_preprocess(np.array(cross_feats, dtype=np.float64), params)
        if adjust:
            Xp = _apply_feature_weights(Xp, feature_names, adjust.get("featureWeights") or {})
        preds = model.predict(Xp).tolist()

        # 人造模型规则过滤：仅保留满足规则的截面样本（与 score_latest 对齐）
        model_rule = getattr(model, 'rule', '')
        if model_rule:
            keep_idx = _apply_rule_filter(
                [{}] * len(cross_codes),
                np.array(cross_feats, dtype=np.float64),
                feature_names, model_rule)
            if len(keep_idx) < len(cross_codes):
                cross_codes = [cross_codes[i] for i in keep_idx]
                cross_feats = [cross_feats[i] for i in keep_idx]
                cross_rets = [cross_rets[i] for i in keep_idx]
                ic_rets = [ic_rets[i] for i in keep_idx]
                preds = [preds[i] for i in keep_idx]
        if len(cross_codes) < groups * 2:
            continue

        ic = _pearson(preds, ic_rets)
        rank_ic = _spearman(preds, ic_rets)
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

        rebalance_idx += 1
        if progress_cb and rebalance_idx % 10 == 0:
            progress_cb(rebalance_idx, (total_dates // step) + 1)

    if not ic_series:
        raise ValueError(
            f"有效截面样本不足，无法完成 ML 信号分层回测：候选池 {len(pool)} 只、有效 {len(series)} 只，"
            f"但所有调仓日截面均未凑够 {groups * 2} 只（多因停牌/涨跌停/因子缺失被跳过）。"
            f"请增大候选池规模、缩短持有期(n)或放宽板块范围。"
        )

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

    result = {
        "factorLabel": f"ML模型({mid})", "groups": groups, "n": n, "modelId": mid,
        "meanIc": ic_stats["meanIc"], "meanRankIc": mean_rank_ic,
        "icWinRate": ic_stats["icWinRate"], "icIr": ic_stats["icIr"],
        "icSeries": ic_series, "groupSummary": group_summary, "longShort": long_short_points,
        "metrics": metrics, "benchmark": bench_metrics,
        "universeSize": len(pool), "effectiveStocks": len(series), "rebalanceCount": len(ic_series),
        "survivorshipBiasWarning": "候选池为当前上市股票快照，已退市股票不在回测池中，历史收益可能系统性高估",
    }
    if config is not None:
        result["config"] = config
    # 快照因子前视风险警告
    if snap_keys:
        snap_labels = []
        for k in snap_keys:
            meta = SNAPSHOT_FACTORS.get(k, {})
            snap_labels.append(meta.get("label", k))
        snap_str = "、".join(snap_labels[:8])
        if len(snap_labels) > 8:
            snap_str += f"等{len(snap_labels)}项"
        result["snapshotWarning"] = (
            f"含快照因子：{snap_str}。"
            "当前最新值被应用到回测所有历史截面（beginning-of-sample look-ahead bias），"
            "回测绩效（IC/Sharpe/分组收益）系统性高估，不可直接指导实盘。"
            "快照因子回测仅用于探索因子方向，实盘因子组应全部使用历史时序特征。"
        )
    # 报告存档由路由层负责（带 user_id），此处跳过
    return result


def _equity_curve(returns: list[float]) -> list[float]:
    eq = [1.0]
    for r in returns:
        eq.append(eq[-1] * (1.0 + r))
    return eq


async def score_codes(mid: str, codes: list[str]) -> list[dict]:
    """对指定代码列表用落盘模型打分（供盯盘调度复用，不依赖行情列表接口）。

    快照特征用腾讯行情字段（turnover/pe/pb/市值）填充，财务类字段缺失则跳过该股。
    返回 [{code, score}]，按代码顺序（与输入对齐）。
    """
    bundle = _load_model(mid)
    feature_names = bundle["feature_names"]
    model = bundle["model"]
    params = bundle["preprocess"]
    snap_set = set(SNAPSHOT_FACTORS) - {"sector"}
    snap_keys = [k for k in feature_names if k in snap_set]
    quotes = {}
    if snap_keys:
        try:
            quotes = await adapters.fetch_quotes(codes)
        except Exception:
            quotes = {}
    # 财务/资金流因子需要额外拉取，否则 snapshot_factor_value 返回 None → 丢弃全部
    pool = [{"code": c, **(quotes.get(c) or {})} for c in codes]
    await _enrich_pool_extra(pool, snap_keys)
    enriched = {r["code"]: r for r in pool}
    sem = asyncio.Semaphore(min(50, max(15, len(codes))))
    collected = []

    async def one(code):
        async with sem:
            kline = await adapters.fetch_kline_retry(code, 260)
            if not kline:
                logger.warning("盯盘模型打分拉取K线失败 code=%s", code)
                return
            if len(kline) < 60:
                return
            arr = kline_to_arrays(kline)
            i = len(kline) - 1
            fvals = []
            for k in feature_names:
                if k in snap_set:
                    v = snapshot_factor_value(enriched.get(code) or {}, k)
                else:
                    v = series_at(compute_factor_series(k, arr), i)
                if v is None:
                    return
                fvals.append(v)
            collected.append((code, fvals))

    await asyncio.gather(*(one(c) for c in codes))
    if not collected:
        return []
    X = np.array([f for _, f in collected], dtype=np.float64)
    Xp = _apply_preprocess(X, params)
    preds = model.predict(Xp).tolist()
    # 人造模型规则过滤
    model_rule = getattr(model, 'rule', '')
    if model_rule:
        keep_idx = _apply_rule_filter(
            [{}] * len(collected), X, feature_names, model_rule)
        if len(keep_idx) < len(collected):
            collected = [collected[i] for i in keep_idx]
            preds = [preds[i] for i in keep_idx]
    return [{"code": c, "score": float(p)} for c, p in zip([c for c, _ in collected], preds)]


def import_model_file(filename: str, data: bytes) -> dict:
    """导入外部训练好的模型文件（joblib bundle，需含 model 与 feature_names）。

    写入临时文件加载校验通过后落盘 ml_models/，缺失 preprocess 时用恒等预处理
    （均值0/标准差1）兜底，保证打分/回测/盯盘可推理；随后登记 sidecar JSON 元数据。
    """
    import tempfile
    suffix = os.path.splitext(filename or "model.joblib")[1] or ".joblib"
    if suffix.lower() not in (".joblib", ".pkl", ".pickle"):
        raise ValueError("仅支持 .joblib/.pkl/.pickle 模型文件")
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        bundle = joblib.load(tmp)
    except Exception as e:
        raise ValueError(f"模型文件无法加载: {e}")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    feature_names = bundle.get("feature_names")
    model = bundle.get("model")
    if not feature_names or model is None:
        raise ValueError("模型文件需包含 model 与 feature_names 字段（本平台 train 产物即此格式）")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mid = f"import_{ts}"
    path = os.path.join(ML_DIR, f"{mid}.joblib")
    joblib.dump({
        "model": model,
        "feature_names": list(feature_names),
        "model_type": bundle.get("model_type", "gbdt"),
        # 外部模型可能无预处理参数：用恒等变换兜底（clip±inf/均值0/标准差1 不改变输入）
        "preprocess": bundle.get("preprocess") or {
            "lo": float("-inf"), "hi": float("inf"),
            "mean": 0.0, "std": 1.0,
        },
    }, path)
    meta = {
        "id": mid, "path": path, "modelType": bundle.get("model_type", "gbdt"),
        "featureNames": list(feature_names),
        "imported": True, "sourceFile": filename or "",
        "trainedAt": datetime.datetime.now().isoformat(),
    }
    # 导入时提取重要性写入侧车 JSON，调参接口无需反序列化整包模型
    if model is not None and hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        meta["featureImportance"] = sorted(
            [{"feature": list(feature_names)[i], "importance": float(imp[i])}
             for i in range(min(len(feature_names), len(imp)))],
            key=lambda x: x["importance"], reverse=True)
    try:
        with open(os.path.join(ML_DIR, f"{mid}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return meta


def _alpha_beta(bench: list[float], strat: list[float]) -> dict:
    n = len(bench)
    if n < 2:
        return {"alpha": 0.0, "beta": 0.0}
    mx, my = mean(bench), mean(strat)
    num = sum((bench[i] - mx) * (strat[i] - my) for i in range(n))
    den = sum((b - mx) ** 2 for b in bench)
    b = 0.0 if den == 0 else num / den
    return {"alpha": my - b * mx, "beta": b}


def _parse_rule(rule: str, feature_names: list[str]) -> str | None:
    """将中文规则字符串转为可 eval 的 Python 表达式。

    支持的语法：因子名 > < >= <= 数值，and/or 连接。
    因子名先在 feature_names 中精确匹配，再尝试模糊匹配（key 包含关系）。
    返回可 eval 的表达式字符串，或 None（解析失败）。
    """
    if not rule or not rule.strip():
        return None
    expr = rule.strip()
    # 将中文逻辑运算符替换为 Python 语法
    expr = expr.replace("且", " and ").replace("或", " or ").replace("并且", " and ").replace("或者", " or ")
    expr = expr.replace("大于等于", ">=").replace("小于等于", "<=").replace("大于", ">").replace("小于", "<")
    expr = expr.replace("等于", "==").replace("不等于", "!=").replace("不低于", ">=").replace("不超过", "<=")
    # 将因子名映射为数组索引（按 key 长度降序，避免短 key 先匹配长 key 前缀）
    sorted_fns = sorted(enumerate(feature_names), key=lambda x: -len(x[1]))
    for i, fn in sorted_fns:
        if re.search(r'\b' + re.escape(fn) + r'\b', expr):
            expr = re.sub(r'\b' + re.escape(fn) + r'\b', f"f[{i}]", expr)
    # 还没替换的因子名：尝试模糊匹配中文 label
    for i, fn in sorted_fns:
        label = ((FACTORS.get(fn) or SNAPSHOT_FACTORS.get(fn)) or {}).get("label", "")
        if label and label in expr:
            expr = expr.replace(label, f"f[{i}]")
    # 验证：表达式中不应再有中文（未识别的因子名）
    if re.search(r'[\u4e00-\u9fff]{2,}', expr):
        raise ValueError(f"规则中包含未识别的因子名: {rule}")
    return expr


def _eval_expr_ast(node, f):
    """安全求值规则表达式 AST 节点，仅允许比较/布尔/下标/常量。"""
    import operator
    ops_map = {
        ast.Eq: operator.eq, ast.NotEq: operator.ne,
        ast.Gt: operator.gt, ast.GtE: operator.ge,
        ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.And: lambda a, b: a and b, ast.Or: lambda a, b: a or b,
    }
    if isinstance(node, ast.BoolOp):
        values = [_eval_expr_ast(v, f) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
        raise ValueError(f"不支持的布尔运算符: {type(node.op)}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval_expr_ast(node.operand, f)
    if isinstance(node, ast.Compare):
        left = _eval_expr_ast(node.left, f)
        result = True
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_expr_ast(comparator, f)
            result = result and ops_map[type(op)](left, right)
            left = right
        return result
    if isinstance(node, ast.Subscript):
        idx = node.slice.value if isinstance(node.slice, ast.Constant) else node.slice
        if isinstance(node.value, ast.Name) and node.value.id == "f":
            return float(f[int(idx)])
        raise ValueError(f"不支持的数组名: {node.value.id if isinstance(node.value, ast.Name) else '?'}")
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_expr_ast(node.operand, f)
    raise ValueError(f"不支持的 AST 节点: {ast.dump(node)}")


def _apply_rule_filter(rows: list[dict], fvals_matrix: np.ndarray,
                       feature_names: list[str], rule: str) -> list[int]:
    """按 rule 过滤样本，返回通过过滤的索引列表。rule 为空时不过滤；解析失败抛错。"""
    try:
        expr_str = _parse_rule(rule, feature_names)
    except ValueError:
        raise
    if not expr_str:
        return list(range(len(rows)))
    try:
        tree = ast.parse(expr_str.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"规则表达式语法错误: {rule}") from e
    keep = []
    for i, f in enumerate(fvals_matrix):
        try:
            if _eval_expr_ast(tree.body, f):
                keep.append(i)
        except Exception:
            pass  # 单样本求值异常（如除零）→ 不纳入
    return keep


# ---------------- 人造/手动模型（P4） ----------------

MANUAL_FEATURES = sorted(list(FACTORS.keys()) + [k for k in SNAPSHOT_FACTORS if SNAPSHOT_FACTORS[k].get("format") != "categorical"])


def manual_feature_options() -> list[dict]:
    """可选手工构建模型的因子（技术因子 + 快照因子，与 build_dataset 同构，可直接推理）。"""
    options = []
    for k in MANUAL_FEATURES:
        meta = FACTORS.get(k) or SNAPSHOT_FACTORS.get(k) or {}
        options.append({"key": k, "label": meta.get("label", k), "group": meta.get("group", "")})
    return options


class _ManualModel:
    """手动加权线性模型：predict 时对截面特征做 z-score 后再按权重线性合成 + 阈值偏移。

    与训练模型同构（实现 .predict + feature_importances_），可零改动接入
    score_latest / backtest_model / 盯盘调度（score_codes）。
    """

    def __init__(self, feature_names: list[str], weights: dict, threshold: float = 0.0,
                 rule: str = ""):
        self.feature_names = list(feature_names)
        self.weights = [float(weights.get(n, 0.0)) for n in self.feature_names]
        self.threshold = float(threshold or 0.0)
        self.rule = rule or ""

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        if X.shape[0] == 1 or np.all(sd == 0):
            # 单样本/常量列：不 zscore，直接加权
            return X @ np.array(self.weights) + self.threshold
        sd[sd == 0] = 1.0
        Z = (X - mu) / sd
        return Z @ np.array(self.weights) + self.threshold

    @property
    def feature_importances_(self):
        return np.abs(np.array(self.weights))


def create_manual_model(name: str, weights: dict, threshold: float | None = None,
                        rule: str = "") -> dict:
    """创建并落盘一个人造/手动模型（不依赖任何自动训练）。

    feature_names 为全部候选因子（技术 + 快照），落盘结构与自动训练产物完全同构，
    之后即可用于打分、回测、盯盘调度。未指定权重的因子默认权重为 0。返回模型元数据。
    """
    feature_names = MANUAL_FEATURES
    valid = {k: v for k, v in (weights or {}).items() if k in feature_names}
    if not valid:
        raise ValueError("请至少为一个有效因子设置权重")
    model = _ManualModel(feature_names, valid, threshold, rule)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mid = f"manual_{ts}"
    path = os.path.join(ML_DIR, f"{mid}.joblib")
    joblib.dump({
        "model": model,
        "feature_names": feature_names,
        "model_type": "manual",
        # 恒等预处理：手动模型在 predict 内部自行做截面标准化
        "preprocess": {
            "lo": float("-inf"), "hi": float("inf"),
            "mean": 0.0, "std": 1.0,
        },
    }, path)
    meta = {
        "id": mid, "path": path, "modelType": "manual",
        "name": (name or mid).strip(), "featureNames": feature_names,
        "featureWeights": valid, "threshold": float(threshold or 0.0),
        "rule": rule or "", "manual": True,
        "featureImportance": sorted(
            [{"feature": k, "importance": abs(float(v))} for k, v in valid.items()],
            key=lambda x: x["importance"], reverse=True),
        "trainedAt": datetime.datetime.now().isoformat(),
    }
    try:
        with open(os.path.join(ML_DIR, f"{mid}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return meta


def model_import_template() -> dict:
    """外部模型导入引导：返回平台特征清单与示例打包代码（P8 降低导入门槛）。"""
    sample = (
        "import joblib, numpy as np\n"
        "from sklearn.ensemble import GradientBoostingRegressor\n"
        f"feature_names = {MANUAL_FEATURES!r}\n"
        "model = GradientBoostingRegressor(n_estimators=100)\n"
        "X = np.random.rand(200, len(feature_names)); y = X[:, 0] * 0.01\n"
        "model.fit(X, y)\n"
        "joblib.dump({'model': model, 'feature_names': feature_names}, 'my_model.joblib')\n"
    )
    return {
        "featureNames": MANUAL_FEATURES,
        "featureLabels": [{"key": k, "label": (FACTORS.get(k) or SNAPSHOT_FACTORS.get(k) or {}).get("label", k)} for k in MANUAL_FEATURES],
        "sampleCode": sample,
        "note": "模型包需包含 model(实现.predict) 与 feature_names(与本清单一致)；preprocess 缺失时按恒等变换兜底。",
    }
