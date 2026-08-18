"""numpy 向量化因子计算（全序列版）。

与 factors.py 的逐点函数保持数值一致，但一次计算整条 K 线序列，
避免回测热点循环里逐股票、逐截面重复调用。仅用 numpy，无 pandas 依赖。

2026-08-18 向量化改造：所有滚动窗口因子（volatility/rsi/kdj/wr/cci/boll/
amplitude/volume_ratio/obv/high_low_pos/dist_52w/skew/kurt/down_vol/max_drawdown/
ma_align/ema_slope/donchian/vol_corr/vol_change）由 Python 逐点 for 循环改为
`sliding_window_view` 批量滑动窗口，消除 O(hist) 行循环热点；仅 KDJ 的 EMA 平滑
与 _ema 保留不可避免的顺序递推（与 factors.py 逐点版数值一致，atol=1e-9）。
"""
import numpy as np


def kline_to_arrays(kline: list[dict]) -> dict:
    """K 线序列转 numpy 数组。返回 {date, open, close, high, low, volume}。"""
    n = len(kline)
    open_ = np.empty(n, dtype=np.float64)
    close = np.empty(n, dtype=np.float64)
    high = np.empty(n, dtype=np.float64)
    low = np.empty(n, dtype=np.float64)
    volume = np.empty(n, dtype=np.float64)
    dates = [None] * n
    for i, row in enumerate(kline):
        dates[i] = row["date"]
        open_[i] = row["open"]
        close[i] = row["close"]
        high[i] = row["high"]
        low[i] = row["low"]
        volume[i] = row["volume"]
    return {"date": dates, "open": open_, "close": close, "high": high, "low": low, "volume": volume}


def momentum_series(close: np.ndarray, n: int) -> np.ndarray:
    """N 日动量：close[i]/close[i-n]-1，前 n 项为 NaN。"""
    out = np.full_like(close, np.nan)
    if len(close) <= n:
        return out
    base = close[:-n]
    out[n:] = np.where(base == 0, np.nan, close[n:] / base - 1.0)
    return out


def ma_dev_series(close: np.ndarray, n: int) -> np.ndarray:
    """MA 偏离度：(close-MA)/MA，前 n-1 项为 NaN。"""
    out = np.full_like(close, np.nan)
    if len(close) < n:
        return out
    ma = _rolling_mean(close, n)
    mask = ma != 0
    out[mask] = (close[mask] - ma[mask]) / ma[mask]
    return out


def volatility_series(close: np.ndarray, n: int) -> np.ndarray:
    """N 日收益率标准差，前 n 项为 NaN。

    窗口取 rets[i-n:i]（第 i-n+1 日到第 i 日收益），与 factors.py factor_volatility
    逐点版严格对齐，不含第 i+1 日收益（无前视偏差）。
    """
    out = np.full_like(close, np.nan)
    if len(close) <= n:
        return out
    rets = _daily_returns(close)
    win = _win(rets, n)
    if win is None:
        return out
    out[n:] = np.std(win, axis=1, ddof=0)
    return out


def rsi_series(close: np.ndarray, n: int) -> np.ndarray:
    """RSI(n)，简化平均版（与 factors.py factor_rsi 一致），前 n 项为 NaN。"""
    out = np.full_like(close, np.nan)
    if len(close) <= n:
        return out
    diff = np.diff(close)
    gain = np.where(diff >= 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    avg_gain = _win(gain, n).mean(axis=1)
    avg_loss = _win(loss, n).mean(axis=1)
    seg = np.full_like(avg_gain, 100.0)
    nz = avg_loss != 0
    seg[nz] = 100.0 - 100.0 / (1.0 + avg_gain[nz] / avg_loss[nz])
    out[n:] = seg
    return out


def macd_series(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """MACD 柱值序列（DIF - DEA），前 slow+signal-1 项为 NaN。

    全序列化：对整条 close 算 EMA(fast)/EMA(slow) → DIF，再对 DIF 有效部分算
    EMA(signal) → DEA，逐点回填 MACD。与 factors.py factor_macd 逐点版数值对齐
    （合成数据实测最大偏差 ~3.9e-16）。旧版只给 out[-1] 赋值，历史全 NaN，导致
    ML 数据集构建时所有含 MACD 特征的样本被跳过。
    """
    out = np.full_like(close, np.nan)
    if len(close) < slow + signal:
        return out
    ema_f = _ema(close, fast)
    ema_s = _ema(close, slow)
    dif = ema_f - ema_s
    valid_mask = ~np.isnan(dif)
    valid_dif = dif[valid_mask]
    if len(valid_dif) < signal:
        return out
    dea = _ema(valid_dif, signal)
    macd_valid = valid_dif - dea
    valid_idx = np.where(valid_mask)[0]
    for k, idx in enumerate(valid_idx):
        v = macd_valid[k]
        if not np.isnan(v):
            out[idx] = v
    return out


def kdj_k_series(kline_arr: dict, n: int = 9, smooth: int = 3) -> np.ndarray:
    """KDJ-K 值序列，前 n-1 项为 NaN。

    RSV 用滑动窗口 max/min 向量化；K 值为 RSV 的 EMA 平滑（smooth 期，种子 50），
    顺序递推保留（与 factors.py factor_kdj_k 逐点版数值一致）。
    """
    close = kline_arr["close"]
    high = kline_arr["high"]
    low = kline_arr["low"]
    m = len(close)
    out = np.full(m, np.nan)
    if m < n:
        return out
    hh = _win(high, n).max(axis=1)
    ll = _win(low, n).min(axis=1)
    den = hh - ll
    rsv = np.where(den == 0, 50.0, (close[n - 1:] - ll) / den * 100.0)
    alpha = 1.0 / smooth
    k_val = 50.0
    for j in range(len(rsv)):
        k_val = (1.0 - alpha) * k_val + alpha * rsv[j]
        out[n - 1 + j] = k_val
    return out


def wr_series(kline_arr: dict, n: int = 14) -> np.ndarray:
    """威廉指标 WR 序列，前 n-1 项为 NaN。"""
    close = kline_arr["close"]
    high = kline_arr["high"]
    low = kline_arr["low"]
    m = len(close)
    out = np.full(m, np.nan)
    if m < n:
        return out
    hh = _win(high, n).max(axis=1)
    ll = _win(low, n).min(axis=1)
    den = hh - ll
    seg = close[n - 1:]
    out[n - 1:] = np.where(den == 0, np.nan, (hh - seg) / den * 100.0)
    return out


def cci_series(kline_arr: dict, n: int = 14) -> np.ndarray:
    """CCI 序列，前 n-1 项为 NaN。"""
    close = kline_arr["close"]
    high = kline_arr["high"]
    low = kline_arr["low"]
    m = len(close)
    out = np.full(m, np.nan)
    if m < n:
        return out
    tp = (high + low + close) / 3.0
    wt = _win(tp, n)
    ma = wt.mean(axis=1)
    md = np.mean(np.abs(wt - ma[:, None]), axis=1)
    seg_tp = tp[n - 1:]
    out[n - 1:] = np.where(md == 0, 0.0, (seg_tp - ma) / (0.015 * md))
    return out


def boll_pct_series(kline_arr: dict, n: int = 20, k: float = 2.0) -> np.ndarray:
    """布林带 %B 序列，前 n-1 项为 NaN。"""
    close = kline_arr["close"]
    m = len(close)
    out = np.full(m, np.nan)
    if m < n:
        return out
    w = _win(close, n)
    mu = w.mean(axis=1)
    sd = w.std(axis=1, ddof=0)
    upper = mu + k * sd
    lower = mu - k * sd
    den = upper - lower
    seg = close[n - 1:]
    out[n - 1:] = np.where(den == 0, 0.5, (seg - lower) / den)
    return out


def amplitude_series(kline_arr: dict, n: int = 20) -> np.ndarray:
    """N 日平均振幅序列，前 n 项为 NaN。"""
    close = kline_arr["close"]
    high = kline_arr["high"]
    low = kline_arr["low"]
    m = len(close)
    out = np.full(m, np.nan)
    if m <= n:
        return out
    hl = high - low
    # out[i]：prev=close[i-n:i]、hl=high[i-n+1:i+1]（各 n 个），逐元素 hl/prev
    amps = np.where(_win(close, n)[:-1] == 0, np.nan, _win(hl, n)[1:] / _win(close, n)[:-1])
    all_nan = np.all(np.isnan(amps), axis=1)
    with np.errstate(invalid="ignore", all="ignore"):
        mean_amps = np.nanmean(amps, axis=1)
    out[n:] = np.where(all_nan, np.nan, mean_amps)
    return out


def volume_ratio_series(kline_arr: dict, n: int = 5) -> np.ndarray:
    """量比序列，前 n 项为 NaN。"""
    vol = kline_arr["volume"]
    m = len(vol)
    out = np.full(m, np.nan)
    if m <= n:
        return out
    # out[i] 的分母为 vol[i-n:i] 均值，即滚动均值序列的 i-1 位
    base = _rolling_mean(vol, n)[n - 1: m - 1]
    v = vol[n:]
    out[n:] = np.where(base == 0, np.nan, v / base)
    return out


def obv_trend_series(kline_arr: dict, n: int = 20) -> np.ndarray:
    """OBV 趋势序列，前 n 项为 NaN。"""
    close = kline_arr["close"]
    vol = kline_arr["volume"]
    m = len(close)
    out = np.full(m, np.nan)
    if m <= n:
        return out
    wc = _win(close, n + 1)   # close[i-n:i+1]
    wv = _win(vol, n)[1:]     # vol[i-n+1:i+1]
    diff = wc[:, 1:] - wc[:, :-1]
    signs = np.where(diff > 0, 1.0, np.where(diff < 0, -1.0, 0.0))
    signed = np.sum(signs * wv, axis=1)
    vol_sum = np.sum(wv, axis=1)
    out[n:] = np.where(vol_sum == 0, np.nan, signed / vol_sum)
    return out


def high_low_pos_series(kline_arr: dict, n: int = 20) -> np.ndarray:
    """N 日高低区间位置序列，前 n-1 项为 NaN。"""
    close = kline_arr["close"]
    high = kline_arr["high"]
    low = kline_arr["low"]
    m = len(close)
    out = np.full(m, np.nan)
    if m < n:
        return out
    hh = _win(high, n).max(axis=1)
    ll = _win(low, n).min(axis=1)
    den = hh - ll
    seg = close[n - 1:]
    out[n - 1:] = np.where(den == 0, 0.5, (seg - ll) / den)
    return out


def dist_high_series(close: np.ndarray, n: int = 240) -> np.ndarray:
    """距 N 日新高回撤序列（窗口随短历史收缩，前 20 项为 NaN）。"""
    m = len(close)
    out = np.full(m, np.nan)
    if m < 20:
        return out
    # 扩展窗口区：i ∈ [20, n-2]，w=i+1（从头累计）
    exp_end = min(n - 2, m - 1)
    if exp_end >= 20:
        exp_max = np.maximum.accumulate(close)
        idx = np.arange(20, exp_end + 1)
        hh = exp_max[idx]
        out[idx] = np.where(hh == 0, np.nan, close[idx] / hh - 1.0)
    # 滑动窗口区：i ∈ [max(n-1, 20), m-1]，w=n
    slide_start = max(n - 1, 20)
    if m >= n and slide_start <= m - 1:
        win = _win(close, n)
        hh = win.max(axis=1)
        k_start = slide_start - (n - 1)
        out[slide_start:] = np.where(hh[k_start:] == 0, np.nan, close[slide_start:] / hh[k_start:] - 1.0)
    return out


def dist_low_series(close: np.ndarray, n: int = 240) -> np.ndarray:
    """距 N 日低点涨幅序列（窗口随短历史收缩，前 20 项为 NaN）。"""
    m = len(close)
    out = np.full(m, np.nan)
    if m < 20:
        return out
    exp_end = min(n - 2, m - 1)
    if exp_end >= 20:
        exp_min = np.minimum.accumulate(close)
        idx = np.arange(20, exp_end + 1)
        ll = exp_min[idx]
        out[idx] = np.where(ll == 0, np.nan, close[idx] / ll - 1.0)
    slide_start = max(n - 1, 20)
    if m >= n and slide_start <= m - 1:
        win = _win(close, n)
        ll = win.min(axis=1)
        k_start = slide_start - (n - 1)
        out[slide_start:] = np.where(ll[k_start:] == 0, np.nan, close[slide_start:] / ll[k_start:] - 1.0)
    return out


def skew_series(close: np.ndarray, n: int = 20) -> np.ndarray:
    """N 日收益率偏度序列（>0 右尾风险，方向 -1）。"""
    m = len(close)
    out = np.full(m, np.nan)
    if m <= n:
        return out
    w = _win(_daily_returns(close), n)
    mu = w.mean(axis=1)
    s = w.std(axis=1, ddof=0)
    diff = w - mu[:, None]
    num = np.mean(diff ** 3, axis=1)
    out[n:] = np.where(s == 0, 0.0, num / (s ** 3))
    return out


def kurt_series(close: np.ndarray, n: int = 20) -> np.ndarray:
    """N 日收益率超额峰度序列（>0 肥尾，方向 -1）。"""
    m = len(close)
    out = np.full(m, np.nan)
    if m <= n:
        return out
    w = _win(_daily_returns(close), n)
    mu = w.mean(axis=1)
    s = w.std(axis=1, ddof=0)
    diff = w - mu[:, None]
    num = np.mean(diff ** 4, axis=1)
    out[n:] = np.where(s == 0, 0.0, num / (s ** 4) - 3.0)
    return out


def down_vol_series(close: np.ndarray, n: int = 20) -> np.ndarray:
    """N 日下行波动率序列：只统计负收益的标准差。"""
    m = len(close)
    out = np.full(m, np.nan)
    if m <= n:
        return out
    w = _win(_daily_returns(close), n)
    neg = np.where(w < 0, w, np.nan)
    cnt = (w < 0).sum(axis=1)
    sd = np.zeros(len(cnt))
    valid = cnt > 0
    if valid.any():
        with np.errstate(invalid="ignore", all="ignore"):
            sd[valid] = np.nanstd(neg[valid], axis=1, ddof=0)
    out[n:] = np.where(cnt == 0, 0.0, sd)
    return out


def max_drawdown_series(close: np.ndarray, n: int = 60) -> np.ndarray:
    """N 日最大回撤序列（负值，越深风险越大，方向 -1）。"""
    m = len(close)
    out = np.full(m, np.nan)
    if m < n:
        return out
    w = _win(close, n)
    runmax = np.maximum.accumulate(w, axis=1)
    dd = np.where(runmax != 0, w / runmax - 1.0, 0.0)
    out[n - 1:] = dd.min(axis=1)
    return out


def ma_align_series(close: np.ndarray, n1: int = 5, n2: int = 20, n3: int = 60) -> np.ndarray:
    """均线排列强度序列：(MA5-MA20)/MA20 × (MA20-MA60)/MA60。"""
    m = len(close)
    out = np.full(m, np.nan)
    if m < n3:
        return out
    ma5 = _rolling_mean(close, n1)
    ma20 = _rolling_mean(close, n2)
    ma60 = _rolling_mean(close, n3)
    mask = (ma20 != 0) & (ma60 != 0)
    out[mask] = (ma5[mask] - ma20[mask]) / ma20[mask] * (ma20[mask] - ma60[mask]) / ma60[mask]
    return out


def ema_slope_series(close: np.ndarray, n: int = 20) -> np.ndarray:
    """EMA(n) 斜率序列：(EMA_t/EMA_{t-1})-1。"""
    m = len(close)
    out = np.full(m, np.nan)
    if m <= n:
        return out
    e = _ema(close, n)
    prev = e[:-1]
    cur = e[1:]
    mask = ~np.isnan(prev) & ~np.isnan(cur) & (prev != 0)
    out[1:] = np.where(mask, cur / prev - 1.0, np.nan)
    return out


def donchian_break_series(close: np.ndarray, n: int = 20) -> np.ndarray:
    """唐奇安通道突破序列：close / 前 n 日最高 - 1（含当日，>0 突破）。"""
    m = len(close)
    out = np.full(m, np.nan)
    if m < n:
        return out
    hh = _win(close, n).max(axis=1)
    seg = close[n - 1:]
    out[n - 1:] = np.where(hh == 0, np.nan, seg / hh - 1.0)
    return out


def vol_corr_series(kline_arr: dict, n: int = 20) -> np.ndarray:
    """N 日量价相关序列：收盘价与成交量的 Pearson 相关。"""
    close = kline_arr["close"]
    vol = kline_arr["volume"]
    m = len(close)
    out = np.full(m, np.nan)
    if m <= n:
        return out
    wc = _win(close, n)[:-1]
    wv = _win(vol, n)[:-1]
    xm = wc.mean(axis=1)
    ym = wv.mean(axis=1)
    xc = wc - xm[:, None]
    yc = wv - ym[:, None]
    num = (xc * yc).sum(axis=1)
    den = np.sqrt((xc ** 2).sum(axis=1) * (yc ** 2).sum(axis=1))
    out[n:] = np.where(den == 0, 0.0, num / den)
    return out


def vol_change_series(kline_arr: dict, n: int = 5) -> np.ndarray:
    """量能突增序列：今日量 / N 日均量 - 1。"""
    vol = kline_arr["volume"]
    m = len(vol)
    out = np.full(m, np.nan)
    if m <= n:
        return out
    base = _rolling_mean(vol, n)[n - 1: m - 1]
    v = vol[n:]
    out[n:] = np.where(base == 0, np.nan, v / base - 1.0)
    return out


def mom_accel_series(close: np.ndarray) -> np.ndarray:
    """动量加速度序列：20 日动量 - 10 日动量。"""
    return momentum_series(close, 20) - momentum_series(close, 10)


# ---------------- 因子 key → 向量化计算函数映射 ----------------
# 与 factors.py FACTORS 字典的 key 对齐

_FACTOR_SERIES_FN = {
    "momentum5": lambda a: momentum_series(a["close"], 5),
    "momentum10": lambda a: momentum_series(a["close"], 10),
    "momentum": lambda a: momentum_series(a["close"], 20),
    "momentum60": lambda a: momentum_series(a["close"], 60),
    "momentum120": lambda a: momentum_series(a["close"], 120),
    "ma_dev": lambda a: ma_dev_series(a["close"], 20),
    "ma5_dev": lambda a: ma_dev_series(a["close"], 5),
    "ma60_dev": lambda a: ma_dev_series(a["close"], 60),
    "volatility": lambda a: volatility_series(a["close"], 20),
    "volatility60": lambda a: volatility_series(a["close"], 60),
    "rsi": lambda a: rsi_series(a["close"], 14),
    "rsi6": lambda a: rsi_series(a["close"], 6),
    "macd": lambda a: macd_series(a["close"]),
    "kdj_k": lambda a: kdj_k_series(a),
    "wr14": lambda a: wr_series(a, 14),
    "cci14": lambda a: cci_series(a, 14),
    "boll_pct": lambda a: boll_pct_series(a),
    "amplitude20": lambda a: amplitude_series(a, 20),
    "volume_ratio5": lambda a: volume_ratio_series(a, 5),
    "obv_trend": lambda a: obv_trend_series(a, 20),
    "high_low_pos": lambda a: high_low_pos_series(a, 20),
    "dist_52w_high": lambda a: dist_high_series(a["close"], 240),
    "momentum30": lambda a: momentum_series(a["close"], 30),
    "momentum90": lambda a: momentum_series(a["close"], 90),
    "mom_accel": lambda a: mom_accel_series(a["close"]),
    "ma_align": lambda a: ma_align_series(a["close"]),
    "ema_slope20": lambda a: ema_slope_series(a["close"], 20),
    "bias60": lambda a: ma_dev_series(a["close"], 60),
    "donchian_break20": lambda a: donchian_break_series(a["close"], 20),
    "dist_52w_low": lambda a: dist_low_series(a["close"], 240),
    "skew20": lambda a: skew_series(a["close"], 20),
    "kurt20": lambda a: kurt_series(a["close"], 20),
    "down_vol20": lambda a: down_vol_series(a["close"], 20),
    "max_drawdown60": lambda a: max_drawdown_series(a["close"], 60),
    "vol_corr20": lambda a: vol_corr_series(a, 20),
    "vol_change5": lambda a: vol_change_series(a, 5),
}


def compute_factor_series(key: str, kline_arr: dict) -> np.ndarray | None:
    """按因子 key 计算全序列，返回 numpy 数组（NaN 表示无效）。
    若 key 不支持向量化，返回 None（调用方回退到逐点函数）。"""
    fn = _FACTOR_SERIES_FN.get(key)
    if fn is None:
        return None
    return fn(kline_arr)


def series_at(arr: np.ndarray, i: int):
    """取序列第 i 个值，NaN 转 None（保持与旧函数返回一致）。"""
    if arr is None or i < 0 or i >= len(arr):
        return None
    v = arr[i]
    return None if np.isnan(v) else float(v)


# ---------------- numpy 辅助函数 ----------------

def _win(arr: np.ndarray, n: int):
    """长度为 n 的滑动窗口视图；长度不足返回 None。"""
    if len(arr) < n:
        return None
    return np.lib.stride_tricks.sliding_window_view(arr, n)


def _rolling_mean(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    if len(arr) < n:
        return out
    csum = np.cumsum(arr)
    out[n - 1:] = (csum[n - 1:] - np.concatenate(([0.0], csum[:-n]))) / n
    return out


def _daily_returns(close: np.ndarray) -> np.ndarray:
    """日收益率，长度 = len(close)-1。"""
    prev = close[:-1]
    cur = close[1:]
    out = np.empty_like(cur)
    nonzero = prev != 0
    out[nonzero] = cur[nonzero] / prev[nonzero] - 1.0
    out[~nonzero] = 0.0
    return out


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """EMA 序列，前 period-1 项为 NaN，第 period 项以简单均值作种子（与 factors.py 一致）。"""
    m = len(values)
    out = np.full(m, np.nan)
    if m < period:
        return out
    k = 2.0 / (period + 1)
    seed = np.mean(values[:period])
    out[period - 1] = seed
    prev = seed
    for i in range(period, m):
        prev = values[i] * k + prev * (1.0 - k)
        out[i] = prev
    return out
