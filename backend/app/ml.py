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


async def _kline_window_retry(code: str, end_date: str, days: int,
                              attempts: int = 3, delay: float = 0.5) -> list[dict]:
    """K线窗口拉取（终止于 end_date）带指数退避重试。空结果也重试。"""
    last_error = None
    for attempt in range(attempts):
        try:
            kl = await adapters.fetch_kline_window(code, end_date, days)
            if kl:
                return kl
        except Exception as e:
            last_error = e
        if attempt < attempts - 1:
            await asyncio.sleep(delay * (2 ** attempt))
    if last_error:
        logger.warning("K线窗口拉取失败 code=%s end=%s days=%s error=%s", code, end_date, days, last_error)
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
    # 时间段窗口适配：指定的 start_date/end_date 需要 hist 覆盖回测区间，
    # 否则 kline 拉取的时间范围与过滤区间无交集 → 必然「有效样本不足」
    if start_date or end_date:
        import datetime as _dt
        today = _dt.date.today()
        from_date = _dt.date.fromisoformat(start_date) if (isinstance(start_date, str) and start_date) else None
        to_date = _dt.date.fromisoformat(end_date) if (isinstance(end_date, str) and end_date) else None
        if from_date and to_date:
            if to_date < from_date:
                raise ValueError(f"结束日({end_date})早于起始日({start_date})，请检查时间段设置")
            # 两端都填：窗口为 [from, to]，hist 只需覆盖区间跨度 + 因子回看；
            # today→to_date 的 gap 由 fetch_kline_window 内部自行处理，此处再加会重复
            days_to_cover = (to_date - from_date).days + 60 + n + _MAX_FACTOR_LOOKBACK
        elif from_date:
            # 仅 start：从今天往前抓，需覆盖 today→from_date 再留因子回看
            days_to_cover = (today - from_date).days + 60 + n + _MAX_FACTOR_LOOKBACK
        elif to_date and to_date < today:
            # 仅 end：窗口终止于 to_date，hist 表示 to_date 之前的样本量；
            # gap 同样由 fetch_kline_window 自行处理，此处只需基础量
            days_to_cover = 60 + n + _MAX_FACTOR_LOOKBACK
        else:
            days_to_cover = 60 + n + _MAX_FACTOR_LOOKBACK
        if days_to_cover > hist:
            logger.info("hist=%d 不足以覆盖时间段 %s~%s，自动抬升至 %d", hist, start_date or "最早", end_date or "今天", days_to_cover)
            hist = days_to_cover
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
            if end_date and asset_class != "future":
                kline = await _kline_window_retry(code, end_date, hist)
            else:
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
        period_hint = (
            f"指定时间段 {start_date or '最早'} ~ {end_date or '今天'} 无可用样本"
            f"（K线可能未覆盖该时间段，或历史长度 {hist} 日不足/数据源无更早数据），"
            f"请增大历史长度(hist)、缩短时间段或放宽板块范围后再试。"
        ) if (start_date or end_date) else (
            f"请增大候选池规模(poolSize)、加长历史(hist)或放宽板块范围后再试。"
        )
        raise ValueError(
            f"有效样本不足，无法构建数据集：候选池 {stat['total']} 只，K线拉取失败 {stat['kline_fail']} 只，"
            f"历史不足 {60 + n} 日被剔除 {stat['kline_short']} 只，因子缺失切片 {stat['factor_missing']} 个。"
            f"{period_hint}"
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
                f"K线历史覆盖 {hist} 日（截至今天），可能未覆盖到目标时间段。"
                f"请增大 hist（建议 ≥ 时间段跨度 + 260 日），或调整时间段范围，或增大候选池后重试。"
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
        # P0 数据区间回显：前端结果页展示实际生效的数据起止日与历史长度
        "data_start": min(meta_dates) if meta_dates else None,
        "data_end": max(meta_dates) if meta_dates else None,
        # 实际生效历史长度 = 样本覆盖的交易日数（而非请求的 hist 拉取天数）
        "effective_hist_days": len(set(meta_dates)),
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
    per_date_rank_ic = []
    for d in sorted(date_ic):
        items = date_ic[d]
        if len(items) >= 3:
            ps = np.array([p for p, _ in items])
            ys = np.array([y for _, y in items])
            per_date_ic.append(_pearson(ps, ys))
            per_date_rank_ic.append(_spearman(ps, ys))
    overall_ic = float(np.mean(per_date_ic)) if per_date_ic else 0.0
    overall_rank_ic = float(np.mean(per_date_rank_ic)) if per_date_rank_ic else 0.0
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
        "direction": "long_short", "allowShort": True,
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
                    entry["direction"] = meta.get("direction", "long_short")
                    entry["allowShort"] = bool(meta.get("allowShort", True))
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
                  "threshold", "imported", "sourceFile", "featureWeights",
                  "direction", "allowShort"):
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
                  "threshold", "imported", "sourceFile", "featureWeights",
                  "direction", "allowShort"):
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


def _auto_adjust(mid: str, adjust: dict | None) -> dict | None:
    """显式 adjust 优先；否则读取模型 sidecar JSON 中的调参权重自动应用。

    克隆模型（clone_xxx）的 featureWeights/threshold 存于 sidecar JSON，
    推理时若调用方未显式传 adjust，则自动套用，实现「另存新模型即带调参权重」。
    """
    if adjust:
        return adjust
    meta_path = os.path.join(ML_DIR, f"{mid}.json")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    # 手动模型 predict 内部已自行应用权重/阈值，跳过避免双重应用
    if meta.get("manual"):
        return None
    fw = meta.get("featureWeights")
    th = meta.get("threshold")
    if not fw and th is None:
        return None
    return {"featureWeights": fw or {}, "threshold": th}


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


def split_directional_scores(rows: list[dict], direction: str = "long_short",
                             allow_short: bool = True, top_ratio: float = 0.34) -> dict:
    """把降序打分列表切分为 做多候选(头部) / 做空候选(尾部)。

    longList = 最高分 top_ratio 部分；shortList = 最低分 top_ratio 部分
    （仅在 allowShort 且非 long_only 时产出）。scores 保留完整降序列表兼容旧调用。
    """
    rows = list(rows or [])
    n = len(rows)
    k = max(1, int(n * top_ratio)) if n else 0
    if direction == "short_only":
        long_list = []
        short_list = rows[-k:] if allow_short else []
    elif direction == "long_only":
        long_list = rows[:k]
        short_list = []
    else:
        long_list = rows[:k]
        short_list = rows[-k:] if allow_short else []
    return {
        "longList": long_list,
        "shortList": short_list,
        "scores": rows,
        "direction": direction,
        "allowShort": bool(allow_short),
    }


def model_direction(mid: str) -> dict:
    """读取模型的 direction / allowShort（无 sidecar 时回退默认 long_short / True）。"""
    meta = load_model_meta(mid) or {}
    direction = (meta.get("direction") or "long_short").lower()
    if direction not in ("long_short", "long_only", "short_only"):
        direction = "long_short"
    return {"direction": direction, "allowShort": bool(meta.get("allowShort", True))}


async def score_latest(mid: str, board: str = "all", pool_size: int = 100,
                       progress_cb=None, adjust: dict | None = None,
                       asset_class: str = "a-share",
                       boards: list[str] | None = None) -> list[dict]:
    """加载落盘模型对候选池最新截面打分，返回按预测分降序的打分列表。"""
    bundle = _load_model(mid)
    adjust = _auto_adjust(mid, adjust)
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
                         boards: list[str] | None = None,
                         direction: str | None = None,
                         include_delisted: bool = True) -> dict:
    """用落盘模型预测分作为截面信号做分层回测，响应结构与 /api/select/backtest 一致，
    前端图表零成本复用。每个调仓日对每只股票取当日因子特征→模型预测分→分层。

    direction: long_short（多空对冲）/ long_only（仅多头）/ short_only（仅空头）。
    不传时回退为模型 sidecar 声明的 direction（无则 long_short）。
    """
    bundle = _load_model(mid)
    if direction is None:
        direction = model_direction(mid)["direction"]
    direction = direction.lower()
    if direction not in ("long_short", "long_only", "short_only"):
        raise ValueError(f"未知交易方向: {direction}（应为 long_short / long_only / short_only）")
    adjust = _auto_adjust(mid, adjust)
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
    # 超大窗口由 adapters.fetch_kline 内部东财分页自动处理
    user_hist = int(hist)
    hist = max(user_hist, min_hist_for_ml(n))
    # 时间段窗口适配：同时考虑 start_date 与 end_date，确保抓取窗口覆盖回测区间
    if start_date or end_date:
        import datetime as _dt
        today = _dt.date.today()
        from_date = _dt.date.fromisoformat(start_date) if (isinstance(start_date, str) and start_date) else None
        to_date = _dt.date.fromisoformat(end_date) if (isinstance(end_date, str) and end_date) else None
        if from_date and to_date:
            if to_date < from_date:
                raise ValueError(f"回测结束日({end_date})早于起始日({start_date})，请检查时间段设置")
            # 两端都填：窗口为 [from, to]，hist 只需覆盖区间跨度 + 因子回看；
            # today→to_date 的 gap 由 fetch_kline_window 内部自行处理，此处再加会重复
            days_to_cover = (to_date - from_date).days + 60 + n + _MAX_FACTOR_LOOKBACK
        elif from_date:
            # 仅 start：从今天往前抓，需覆盖 today→from_date 再留因子回看
            days_to_cover = (today - from_date).days + 60 + n + _MAX_FACTOR_LOOKBACK
        elif to_date and to_date < today:
            # 仅 end：窗口终止于 to_date，hist 表示 to_date 之前的样本量；
            # gap 同样由 fetch_kline_window 自行处理，此处只需基础量
            days_to_cover = 60 + n + _MAX_FACTOR_LOOKBACK
        else:
            days_to_cover = 60 + n + _MAX_FACTOR_LOOKBACK
        if days_to_cover > hist:
            logger.info("hist=%d 不足以覆盖回测时间段 %s~%s，自动抬升至 %d", hist, start_date or "最早", end_date or "今天", days_to_cover)
            hist = days_to_cover
    hist_clamped = hist != user_hist

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
    # 生存偏差修复：纳入「上市/退市区间与回测窗口重叠」的退市股（东财无退市股历史，
    # 退市股 K 线走新浪/腾讯保留的历史），使回测样本与真实可得标的一致。
    delisted_codes: set[str] = set()
    delisted_note = None
    if include_delisted and asset_class != "future":
        try:
            dl = await adapters.fetch_delisted_stocks()
        except Exception as e:
            logger.warning("退市股清单获取失败，回退为仅当前上市池: %s", e)
            dl = []
        if dl:
            win_start = start_date or "0000-01-01"
            win_end = end_date or "9999-12-31"
            existing = set(codes)
            for d in dl:
                code = d.get("code")
                if not code or code in existing or code in delisted_codes:
                    continue
                ld = d.get("list_date") or "00000000"
                dd = d.get("delist_date") or "99991231"
                if ld <= win_end and dd >= win_start:
                    delisted_codes.add(code)
                    pool.append({"code": code, "name": d.get("name", ""), "delisted": True})
            if delisted_codes:
                codes = [row["code"] for row in pool]
                delisted_note = f"已纳入 {len(delisted_codes)} 只退市股"
                logger.info("ML回测纳入退市股 %d 只（窗口 %s ~ %s）", len(delisted_codes), win_start, win_end)
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
            if code in delisted_codes:
                # 退市股东财无历史，走新浪/腾讯（保留退市前K线）；上限 1500（新浪单请求上限）
                kline = await _kline_retry(adapters.fetch_kline, code, min(hist + n + 25, 1500))
            elif end_date and asset_class != "future":
                kline = await _kline_window_retry(code, end_date, hist + n + 25)
            else:
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
            if end_date:
                bench_series = await adapters.fetch_kline_window(bench_code, end_date, hist + n + 25)
            else:
                bench_series = await adapters.fetch_kline(bench_code, hist + n + 25)
        except Exception:
            bench_series = None

    # 快照特征：仅对最近 60 个交易日沿用（避免历史截面用今日财报数据→前视）
    # ref_code 取当前上市池中最长 K 线的股票（退市股 K 线止于退市日，不能作快照生效日参照）
    ref_pool = [c for c in series if c not in delisted_codes] or list(series.keys())
    ref_code = max(ref_pool, key=lambda c: len(series[c]))
    ref_dates = [row["date"] for row in series[ref_code]]
    snapshot_cutoff_date = None
    snapshot_auto_start = None
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
            snapshot_auto_start = snapshot_cutoff_date
            logger.info(
                "模型含快照因子且未指定回测起始日，起点自动设为快照生效日 %s（最近60个交易日，避免前视）",
                snapshot_cutoff_date,
            )

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

    # M6 离散买卖规则：模型自带 bullRule/bearRule 时，回测以规则判定多空信号替代分位分组
    bull_rule = getattr(model, 'bull_rule', '') or ''
    bear_rule = getattr(model, 'bear_rule', '') or ''
    bull_fn, bull_err = compile_signal_rule(bull_rule, feature_names) if bull_rule else (None, '')
    bear_fn, bear_err = compile_signal_rule(bear_rule, feature_names) if bear_rule else (None, '')
    if bull_rule and bull_err:
        raise ValueError(f"看多规则不合法: {bull_err}")
    if bear_rule and bear_err:
        raise ValueError(f"看空规则不合法: {bear_err}")
    use_signal_rules = bool(bull_fn or bear_fn)

    ic_series = []
    bucket_returns = [[] for _ in range(groups)]
    bucket_ret_series = [[] for _ in range(groups)]  # B3：每组逐期净值收益
    bucket_equity = [[1.0] for _ in range(groups)]   # B3：每组净值曲线
    bucket_dates: list[str] = []
    stock_contribution: dict[str, float] = defaultdict(float)  # B5：个股累计贡献
    long_short_points = []
    top_group_returns = []
    position_ledger = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()
    bench_by_date = {}
    cum = 1.0
    cum_top = 1.0
    cum_bottom = 1.0
    last_cross = None  # M8 归因：最后一个调仓日截面快照

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

        # M8 归因：保存最后一个调仓日截面（特征原始值 + 预测分），循环结束后构建特征分位归因
        last_cross = {"date": date_t, "codes": list(cross_codes),
                      "feats": [list(f) for f in cross_feats], "preds": list(preds)}

        if use_signal_rules:
            # 规则模式：bullRule→多头，bearRule→空头，其余中性。scorePct 为当日截面分位。
            n_cross = len(cross_codes)
            order_idx = sorted(range(n_cross), key=lambda k: preds[k])
            pct_of = [0.0] * n_cross
            for rank, k in enumerate(order_idx):
                pct_of[k] = (rank + 1) / (n_cross + 1)
            cur_long: set[str] = set()
            cur_short: set[str] = set()
            long_rets: list[float] = []
            short_rets: list[float] = []
            for k in range(n_cross):
                f = cross_feats[k]
                p = pct_of[k]
                if bull_fn and bull_fn(f, p):
                    cur_long.add(cross_codes[k])
                    long_rets.append(cross_rets[k])
                    bucket_returns[-1].append(cross_rets[k])
                elif bear_fn and bear_fn(f, p):  # 互斥：已判多头不再判空
                    cur_short.add(cross_codes[k])
                    short_rets.append(cross_rets[k])
                    bucket_returns[0].append(cross_rets[k])
            top_ret = mean(long_rets) if long_rets else 0.0
            bottom_ret = mean(short_rets) if short_rets else 0.0
        else:
            sorted_preds = sorted(preds)
            day_buckets = [[] for _ in range(groups)]
            day_bucket_codes = [[] for _ in range(groups)]
            for code, fv, fret in zip(cross_codes, preds, cross_rets):
                b = bucket_index(fv, sorted_preds, groups)
                day_buckets[b].append(fret)
                day_bucket_codes[b].append(code)
                bucket_returns[b].append(fret)
            top_ret = mean(day_buckets[-1]) if day_buckets[-1] else 0.0
            bottom_ret = mean(day_buckets[0]) if day_buckets[0] else 0.0
            cur_long = set(day_bucket_codes[-1]) if day_buckets[-1] else set()
            cur_short = set(day_bucket_codes[0]) if day_buckets[0] else set()

        # B3 分组净值：逐期记录每组平均收益 → 净值曲线
        day_bucket_ret = [0.0] * groups
        if use_signal_rules:
            day_bucket_ret[0] = bottom_ret
            day_bucket_ret[-1] = top_ret
        else:
            for b in range(groups):
                day_bucket_ret[b] = mean(day_buckets[b]) if day_buckets[b] else 0.0
        for g in range(groups):
            bucket_ret_series[g].append(day_bucket_ret[g])
            bucket_equity[g].append(bucket_equity[g][-1] * (1.0 + day_bucket_ret[g]))
        bucket_dates.append(date_t)

        # A1 方向裁剪：long_only 仅保留多头、short_only 仅保留空头
        if direction == "long_only":
            cur_short = set()
        elif direction == "short_only":
            cur_long = set()

        # 边缘保底：至少一侧有持仓仍记录（避免仓位图断裂）
        if cur_long or cur_short:
            long_added = sorted(cur_long - prev_long)
            long_removed = sorted(prev_long - cur_long)
            short_added = sorted(cur_short - prev_short)
            short_removed = sorted(prev_short - cur_short)
            turnover = len(long_added) + len(long_removed) + len(short_added) + len(short_removed)
            position_ledger.append({
                "date": date_t,
                "long": sorted(cur_long),
                "short": sorted(cur_short),
                "longReturn": top_ret,
                "shortReturn": bottom_ret,
                # 后端直算的持仓数/换手（报告图直接使用，避免前端每次重算）
                "longCount": len(cur_long),
                "shortCount": len(cur_short),
                "turnover": turnover,
                # 调仓明细：本期相对上期的买入/卖出（供报告「调仓信号」渲染）
                "longAdded": long_added,
                "longRemoved": long_removed,
                "shortAdded": short_added,
                "shortRemoved": short_removed,
            })
            prev_long = cur_long
            prev_short = cur_short

        # B5 个股累计贡献：多头 +收益，空头 -收益（按方向裁剪后的持仓）
        code_ret = dict(zip(cross_codes, cross_rets))
        for c in cur_long:
            stock_contribution[c] += code_ret.get(c, 0.0)
        for c in cur_short:
            stock_contribution[c] -= code_ret.get(c, 0.0)

        # 方向收益：long_short 双份成本；单腿只扣一份
        net_top = top_ret - cost_rate
        net_bottom = -(bottom_ret) - cost_rate  # 空头腿收益正向化
        if direction == "long_only":
            ls_ret = top_ret
            net_ls = net_top
            leg_ok = bool(cur_long)
        elif direction == "short_only":
            ls_ret = -(bottom_ret)
            net_ls = net_bottom
            leg_ok = bool(cur_short)
        else:  # long_short
            ls_ret = top_ret - bottom_ret
            net_ls = ls_ret - 2.0 * cost_rate
            leg_ok = bool(cur_long) and bool(cur_short)
        if leg_ok:
            cum *= (1.0 + net_ls)
            if direction != "short_only":
                cum_top *= (1.0 + net_top)
                top_group_returns.append(net_top)
            if direction != "long_only":
                cum_bottom *= (1.0 + net_bottom)
            long_short_points.append({
                "date": date_t, "longShort": net_ls, "cum": cum - 1.0,
                "topCum": cum_top - 1.0, "bottomCum": cum_bottom - 1.0, "gross": ls_ret,
            })
            if bench_series and bench_date_idx:
                bi = bench_date_idx.get(date_t)
                if bi is not None and bi + n < len(bench_series):
                    bc = [row["close"] for row in bench_series]
                    bench_by_date[date_t] = bc[bi + n] / bc[bi] - 1

        rebalance_idx += 1
        if progress_cb and rebalance_idx % 10 == 0:
            progress_cb(rebalance_idx, (total_dates // step) + 1)

    if not ic_series:
        actual_start = ref_dates[0] if ref_dates else None
        actual_end = ref_dates[-1] if ref_dates else None
        return {
            "ok": False,
            "reason": "NO_VALID_REBALANCE",
            "error": (
                f"回测区间无有效调仓日：候选池 {len(pool)} 只、有效 {len(series)} 只，"
                f"但所有调仓日截面均未凑够 {groups * 2} 只（多因停牌/涨跌停/因子缺失被跳过）。"
            ),
            "hint": (
                f"历史长度 {hist} 日仅覆盖 {actual_start or '?'} ~ {actual_end or '?'}，"
                f"请增大历史长度(hist)、后移终止日(end_date)，或增大候选池/缩短持有期(n)/放宽板块范围。"
            ),
            "requested": {"start": start_date, "end": end_date, "hist": hist},
            "actualWindow": {"start": actual_start, "end": actual_end},
        }

    # 调仓间隔 = 自适应步长 step（≥ n，长历史时自动放大）→ 年化采样数 = 252/step
    # 旧版误用 252/n：step>n 时收益高估 step/n 倍、Sharpe 高估 √(step/n) 倍
    ppy = 252.0 / max(1, step)

    group_summary = []
    for idx in range(groups):
        rets = bucket_ret_series[idx]
        eq = bucket_equity[idx]
        group_summary.append({
            "group": idx + 1,
            "avgReturn": mean(rets) if rets else 0.0,
            "sample": len(bucket_returns[idx]),
            "cumReturn": eq[-1] - 1.0 if eq else 0.0,
            "annualizedReturn": annualized_return(rets, ppy) if rets else 0.0,
            "sharpe": sharpe_ratio(rets, periods_per_year=ppy) if rets else 0.0,
            "maxDrawdown": max_drawdown(eq) if eq else 0.0,
            "equity": [round(v - 1.0, 6) for v in eq],
        })

    ls_returns = [p["longShort"] for p in long_short_points]
    ls_equity = [1.0]
    for r in ls_returns:
        ls_equity.append(ls_equity[-1] * (1.0 + r))
    gross_returns = [p["gross"] for p in long_short_points]

    avg_turnover = mean([p["turnover"] for p in position_ledger]) if position_ledger else 0.0
    gross_sharpe = sharpe_ratio(gross_returns, periods_per_year=ppy)

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
        # B4：换手率 / 成本侵蚀（毛收益 vs 净收益 Sharpe）
        "turnover": avg_turnover,
        "annualizedTurnover": avg_turnover * ppy,
        "grossSharpe": gross_sharpe,
        "costErosion": sharpe_ratio(ls_returns, periods_per_year=ppy) - gross_sharpe,
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
            # ab 即策略相对基准的 alpha/beta（_alpha_beta(bench, strat) 签名）
            "strategyAlpha": ab["alpha"],
            "strategyBeta": ab["beta"],
            **ab,
        }

    ic_stats = information_coefficient_stats([p["ic"] for p in ic_series], ppy)
    mean_rank_ic = mean([p["rankIc"] for p in ic_series])

    effective_start = ic_series[0]["date"] if ic_series else (start_date or ref_dates[60])
    effective_end = ic_series[-1]["date"] if ic_series else (end_date or ref_dates[-1])
    hist_warning = None
    if hist_clamped:
        hist_warning = (
            f"历史长度(hist)已从 {user_hist} 自动调整至 {hist} 日，"
            f"以确保覆盖内置长周期因子所需的最短窗口（≥{min_hist_for_ml(n)}日）。"
            f"回测实际拉取 {hist} 日 K 线，有效调仓区间 {effective_start} ~ {effective_end}。"
        )
    # M8：模型特征贡献（Top-N 重要性，进入报告「特征贡献」卡片）
    feature_importance = []
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            labels = [(FACTORS.get(k) or SNAPSHOT_FACTORS.get(k) or {}).get("label", k)
                      for k in feature_names]
            feature_importance = sorted(
                [{"feature": feature_names[i],
                  "label": labels[i] if i < len(labels) else feature_names[i],
                  "importance": float(imp[i])}
                 for i in range(min(len(feature_names), len(imp)))],
                key=lambda x: x["importance"], reverse=True,
            )
    except Exception:
        feature_importance = []

    # B7 分年度绩效：按自然年聚合 long_short 收益
    yearly: dict[str, list[float]] = {}
    for p in long_short_points:
        yearly.setdefault(p["date"][:4], []).append(p["longShort"])
    yearly_returns = []
    for yr in sorted(yearly):
        rets = yearly[yr]
        cum_yr = 1.0
        for r in rets:
            cum_yr *= (1.0 + r)
        yearly_returns.append({"year": yr, "return": cum_yr - 1.0, "periods": len(rets)})

    # B5 个股累计贡献 Top 30（按绝对贡献排序）
    name_by_code = {r["code"]: r.get("name", "") for r in pool}
    stock_contrib = sorted(
        [{"code": c, "name": name_by_code.get(c, ""), "contribution": v}
         for c, v in stock_contribution.items()],
        key=lambda x: abs(x["contribution"]), reverse=True,
    )[:30]

    result = {
        "factorLabel": f"ML模型({mid})", "groups": groups, "n": n, "modelId": mid,
        "direction": direction,
        "meanIc": ic_stats["meanIc"], "meanRankIc": mean_rank_ic,
        "icWinRate": ic_stats["icWinRate"], "icIr": ic_stats["icIr"],
        "icTStat": ic_stats["tStat"], "icPValue": ic_stats["pValue"],
        "icSeries": ic_series, "groupSummary": group_summary, "longShort": long_short_points,
        "metrics": metrics, "benchmark": bench_metrics,
        "universeSize": len(pool), "effectiveStocks": len(series), "rebalanceCount": len(ic_series),
        "actualHistDays": hist, "effectiveStart": effective_start, "effectiveEnd": effective_end,
        "positionLedger": position_ledger,
        "featureImportance": feature_importance,
        "yearlyReturns": yearly_returns,
        "stockContribution": stock_contrib,
        "bucketDates": bucket_dates,
        "survivorshipBiasWarning": (
            f"已纳入 {len(delisted_codes)} 只退市股（沪市为主），消除部分生存偏差；"
            "深交所退市名单视数据源可用性可能未全覆盖，历史收益仍可能略偏高"
            if delisted_note else
            "候选池为当前上市股票快照，已退市股票不在回测池中，历史收益可能系统性高估"
        ),
    }
    # M6 离散规则信号模式标记 + 规则回显（报告展示）
    if use_signal_rules:
        result["signalMode"] = "rule"
        result["bullRule"] = bull_rule
        result["bearRule"] = bear_rule
    # M8 预测归因：最后一个调仓日的 Top 看多/看空个股特征分位
    result["topAttribution"] = _build_attribution(last_cross, feature_names, top_n=5)
    if hist_warning:
        result["histWarning"] = hist_warning
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
            "快照因子为当前最新值，仅可用于最近 60 个交易日内的截面；"
            "更早历史截面的快照值属前视数据，已自动跳过。"
            "快照因子回测仅用于探索因子方向，实盘因子组应全部使用历史时序特征。"
        )
    if snapshot_auto_start:
        result["snapshotStartNote"] = (
            f"模型含快照因子且未指定回测起始日，起点已自动设为快照生效日 {snapshot_auto_start}"
            f"（最近 60 个交易日），更早历史截面因前视偏差已跳过。"
            f"如需回测完整历史，请改用不含快照因子的模型或明确指定起始日。"
        )
    # 样本内回测标注：落盘模型在含回测区间的历史上训练，回测绩效为 in-sample
    result["inSampleWarning"] = (
        "本回测为样本内（in-sample）：模型在包含回测区间的历史上训练，"
        "回测区间已被模型见过，IC/Sharpe/分组收益系统性高估，不可作为实盘证据。"
        "如需样本外验证，请在训练面板设定早于回测区间的训练时间段后重新训练模型，"
        "或参考评估(CV)的 oosIc/oosRankIc。"
    )
    # 报告存档由路由层负责（带 user_id），此处跳过
    return result


def _equity_curve(returns: list[float]) -> list[float]:
    eq = [1.0]
    for r in returns:
        eq.append(eq[-1] * (1.0 + r))
    return eq


async def score_codes(mid: str, codes: list[str],
                     adjust: dict | None = None,
                     return_features: bool = False) -> list[dict]:
    """对指定代码列表用落盘模型打分（供盯盘调度复用，不依赖行情列表接口）。

    快照特征用腾讯行情字段（turnover/pe/pb/市值）填充，财务类字段缺失则跳过该股。
    adjust 为调参配置 {featureWeights, threshold}，与 score_latest 口径对齐。
    return_features=True 时每条结果附带 features（与 feature_names 对齐的原始因子值），
    供离散买卖规则（bullRule/bearRule）在盯盘侧求值。
    返回 [{code, score}]（return_features=True 时附加 features）。
    """
    bundle = _load_model(mid)
    adjust = _auto_adjust(mid, adjust)
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
    if adjust:
        Xp = _apply_feature_weights(Xp, feature_names, adjust.get("featureWeights") or {})
    preds = model.predict(Xp)
    if adjust:
        preds = _apply_threshold(preds.tolist(), adjust.get("threshold"))
    else:
        preds = preds.tolist()
    # 人造模型规则过滤
    model_rule = getattr(model, 'rule', '')
    if model_rule:
        keep_idx = _apply_rule_filter(
            [{}] * len(collected), X, feature_names, model_rule)
        if len(keep_idx) < len(collected):
            collected = [collected[i] for i in keep_idx]
            preds = [preds[i] for i in keep_idx]
    if return_features:
        return [{"code": c, "score": float(p),
                 "features": [float(v) if v is not None else 0.0 for v in f]}
                for (c, f), p in zip(collected, preds)]
    return [{"code": c, "score": float(p)} for c, p in zip([c for c, _ in collected], preds)]


def get_signal_rules(mid: str) -> dict:
    """加载模型离散买卖规则（bull_rule/bear_rule），编译为可求值函数。

    返回 {featureNames, bullRule, bearRule, bullFn, bearFn, active}。
    无规则时 active=False、bullFn/bearFn 为 None；规则非法时抛 ValueError。
    用于盯盘模型模式的离散信号判定（规则命中优先于分位阈值）。
    """
    bundle = _load_model(mid)
    model = bundle["model"]
    feature_names = list(bundle.get("feature_names") or [])
    bull_rule = (getattr(model, "bull_rule", "") or "").strip()
    bear_rule = (getattr(model, "bear_rule", "") or "").strip()
    bull_fn = bear_fn = None
    if bull_rule:
        bull_fn, err = compile_signal_rule(bull_rule, feature_names)
        if err:
            raise ValueError(f"看多规则不合法: {err}")
    if bear_rule:
        bear_fn, err = compile_signal_rule(bear_rule, feature_names)
        if err:
            raise ValueError(f"看空规则不合法: {err}")
    return {
        "featureNames": feature_names,
        "bullRule": bull_rule,
        "bearRule": bear_rule,
        "bullFn": bull_fn,
        "bearFn": bear_fn,
        "active": bool(bull_fn or bear_fn),
    }


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
        "direction": "long_short", "allowShort": True,
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


def _parse_signal_rule(rule: str, feature_names: list[str]) -> str:
    """将买卖信号规则字符串转为可安全求值的表达式。

    支持变量：scorePct（当日截面分位 0~1）+ 任意 feature_names 因子（原始值，
    别名 vol=volatility）。支持运算：算术 + - * /、比较 > < >= <= == !=、
    布尔 and/or/not、括号、常量。因子名精确匹配后映射为 f[i]，scorePct 映射为 p。
    """
    if not rule or not rule.strip():
        return ""
    expr = rule.strip()
    expr = expr.replace("且", " and ").replace("或", " or ").replace("并且", " and ").replace("或者", " or ")
    expr = expr.replace("大于等于", ">=").replace("小于等于", "<=").replace("大于", ">").replace("小于", "<")
    expr = expr.replace("等于", "==").replace("不等于", "!=").replace("不低于", ">=").replace("不超过", "<=")
    # scorePct → 特殊变量 p
    expr = re.sub(r'\bscorePct\b', 'p', expr)
    # 别名 vol → volatility（文档示例用 vol，因子 key 为 volatility）
    if 'volatility' in feature_names:
        expr = re.sub(r'\bvol\b', 'volatility', expr)
    # 因子名 → f[i]（按 key 长度降序，避免短 key 先匹配长 key 前缀）
    sorted_fns = sorted(enumerate(feature_names), key=lambda x: -len(x[1]))
    for i, fn in sorted_fns:
        if re.search(r'\b' + re.escape(fn) + r'\b', expr):
            expr = re.sub(r'\b' + re.escape(fn) + r'\b', f"f[{i}]", expr)
    if re.search(r'[\u4e00-\u9fff]{2,}', expr):
        raise ValueError(f"规则中包含未识别的变量名: {rule}")
    return expr


def _eval_signal_ast(node, f, p):
    """安全求值信号规则 AST：算术 + 比较 + 布尔 + f[i] 因子 + p(scorePct)。"""
    import operator as _op
    _BIN = {ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul, ast.Div: _op.truediv}
    _CMP = {ast.Eq: _op.eq, ast.NotEq: _op.ne, ast.Gt: _op.gt, ast.GtE: _op.ge,
            ast.Lt: _op.lt, ast.LtE: _op.le}
    if isinstance(node, ast.Expression):
        return _eval_signal_ast(node.body, f, p)
    if isinstance(node, ast.BoolOp):
        vals = [_eval_signal_ast(v, f, p) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return not _eval_signal_ast(node.operand, f, p)
        if isinstance(node.op, ast.USub):
            return -_eval_signal_ast(node.operand, f, p)
        if isinstance(node.op, ast.UAdd):
            return _eval_signal_ast(node.operand, f, p)
        raise ValueError(f"不支持的一元运算符: {type(node.op)}")
    if isinstance(node, ast.BinOp):
        return _BIN[type(node.op)](_eval_signal_ast(node.left, f, p), _eval_signal_ast(node.right, f, p))
    if isinstance(node, ast.Compare):
        left = _eval_signal_ast(node.left, f, p)
        result = True
        for op, comp in zip(node.ops, node.comparators):
            right = _eval_signal_ast(comp, f, p)
            result = result and _CMP[type(op)](left, right)
            left = right
        return result
    if isinstance(node, ast.Subscript):
        idx = node.slice.value if isinstance(node.slice, ast.Constant) else node.slice
        if isinstance(node.value, ast.Name) and node.value.id == "f":
            v = f[int(idx)]
            return 0.0 if v is None else float(v)
        raise ValueError(f"不支持的数组名: {getattr(node.value, 'id', '?')}")
    if isinstance(node, ast.Name):
        if node.id == "p":
            return float(p)
        raise ValueError(f"未知变量: {node.id}")
    if isinstance(node, ast.Constant):
        return node.value
    raise ValueError(f"不支持的 AST 节点: {ast.dump(node)}")


def compile_signal_rule(rule: str, feature_names: list[str]):
    """编译买卖信号规则。返回 (fn, err)，fn(feature_values, score_pct) -> bool。

    feature_values 为与 feature_names 对齐的因子值列表；score_pct 为该股预测分
    在当日截面的分位（0~1）。规则非法时返回 (None, 错误信息)。
    """
    try:
        expr = _parse_signal_rule(rule, feature_names)
    except ValueError as e:
        return None, str(e)
    if not expr:
        return None, "规则为空"
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return None, f"语法错误: {e}"
    _ALLOWED = (ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
                ast.Name, ast.Constant, ast.Subscript, ast.Load,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.UAdd, ast.USub, ast.Not,
                ast.Eq, ast.NotEq, ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.And, ast.Or)
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            return None, f"不允许的语法: {type(node).__name__}"

    def fn(f, p):
        try:
            return bool(_eval_signal_ast(tree, f, p))
        except Exception:
            return False
    return fn, ""


def _build_attribution(last: dict | None, feature_names: list[str], top_n: int = 5) -> dict | None:
    """M8 预测归因：最后一个调仓日截面的 Top 看多/看空个股各特征截面分位。"""
    if not last:
        return None
    codes = last["codes"]
    feats = last["feats"]
    preds = last["preds"]
    n = len(codes)
    m = len(feature_names)
    if not n or not m:
        return None

    def pct_rank(vals):
        idxs = [i for i, v in enumerate(vals) if v is not None]
        order = sorted(idxs, key=lambda i: vals[i])
        out = [None] * len(vals)
        for rank, i in enumerate(order):
            out[i] = (rank + 1) / (len(order) + 1)
        return out

    fpcts = [pct_rank([feats[i][j] for i in range(n)]) for j in range(m)]
    labels = [(FACTORS.get(k) or SNAPSHOT_FACTORS.get(k) or {}).get("label", k)
              for k in feature_names]
    order = sorted(range(n), key=lambda i: -preds[i])

    def build(idx):
        return {
            "code": codes[idx],
            "score": float(preds[idx]),
            "featurePcts": {feature_names[j]: fpcts[j][idx]
                            for j in range(m) if fpcts[j][idx] is not None},
        }

    return {
        "date": last["date"],
        "featureNames": feature_names,
        "featureLabels": labels,
        "longs": [build(i) for i in order[:top_n]],
        "shorts": [build(i) for i in order[-top_n:][::-1]],
    }


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
                 rule: str = "", bull_rule: str = "", bear_rule: str = ""):
        self.feature_names = list(feature_names)
        self.weights = [float(weights.get(n, 0.0)) for n in self.feature_names]
        self.threshold = float(threshold or 0.0)
        self.rule = rule or ""
        self.bull_rule = bull_rule or ""
        self.bear_rule = bear_rule or ""

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
                        rule: str = "", bull_rule: str = "", bear_rule: str = "",
                        direction: str = "long_short", allow_short: bool = True) -> dict:
    """创建并落盘一个人造/手动模型（不依赖任何自动训练）。

    feature_names 为全部候选因子（技术 + 快照），落盘结构与自动训练产物完全同构，
    之后即可用于打分、回测、盯盘调度。未指定权重的因子默认权重为 0。返回模型元数据。
    bull_rule/bear_rule 为离散买卖信号规则（scorePct + 因子名表达式），回测时替代分位分组。
    direction ∈ {long_short, long_only, short_only}，allowShort 控制是否产出空头候选。
    """
    direction = (direction or "long_short").lower()
    if direction not in ("long_short", "long_only", "short_only"):
        raise ValueError(f"未知交易方向: {direction}")
    feature_names = MANUAL_FEATURES
    valid = {k: v for k, v in (weights or {}).items() if k in feature_names}
    if not valid:
        raise ValueError("请至少为一个有效因子设置权重")
    if bull_rule:
        _, err = compile_signal_rule(bull_rule, feature_names)
        if err:
            raise ValueError(f"看多规则不合法: {err}")
    if bear_rule:
        _, err = compile_signal_rule(bear_rule, feature_names)
        if err:
            raise ValueError(f"看空规则不合法: {err}")
    model = _ManualModel(feature_names, valid, threshold, rule, bull_rule, bear_rule)
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
        "rule": rule or "", "bullRule": bull_rule or "", "bearRule": bear_rule or "",
        "manual": True,
        "direction": direction, "allowShort": bool(allow_short),
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


def clone_model_with_adjust(mid: str, feature_weights: dict | None = None,
                            threshold: float | None = None) -> dict:
    """复制原模型并附上调参权重，另存为一个独立新模型。

    新模型与原模型同源（bundle 完整复制，含 model/preprocess/feature_names），
    仅在侧车 JSON 中写入 featureWeights 和 threshold 供后续推理时 apply。
    新模型 ID 为 clone_{原mid}_{timestamp}。
    """
    src_path = os.path.join(ML_DIR, f"{mid}.joblib")
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"原模型不存在: {mid}")
    src_meta = load_model_meta(mid) or {}
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_id = f"clone_{mid}_{ts}"
    new_path = os.path.join(ML_DIR, f"{new_id}.joblib")
    # 完整复制原模型 bundle
    bundle = joblib.load(src_path)
    joblib.dump(bundle, new_path)
    # 写入新侧车 JSON：包含调参权重
    fw = {k: float(v) for k, v in (feature_weights or {}).items()
          if k in (bundle.get("feature_names") or [])}
    _cloned_model = bundle.get("model")
    _bull_rule = getattr(_cloned_model, 'bull_rule', '') or src_meta.get("bullRule", "")
    _bear_rule = getattr(_cloned_model, 'bear_rule', '') or src_meta.get("bearRule", "")
    new_meta = {
        "id": new_id, "path": new_path,
        "modelType": src_meta.get("modelType", bundle.get("model_type", "gbdt")),
        "featureNames": bundle.get("feature_names") or src_meta.get("featureNames") or [],
        "featureImportance": src_meta.get("featureImportance") or [],
        "featureWeights": fw or None,
        "threshold": float(threshold) if threshold is not None else None,
        "bullRule": _bull_rule or None,
        "bearRule": _bear_rule or None,
        "clonedFrom": mid,
        "trainedAt": datetime.datetime.now().isoformat(),
    }
    try:
        with open(os.path.join(ML_DIR, f"{new_id}.json"), "w", encoding="utf-8") as f:
            json.dump(new_meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return new_meta


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
