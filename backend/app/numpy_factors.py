"""numpy 向量化因子计算（全序列版）。

与 factors.py 的逐点函数保持数值一致，但一次计算整条 K 线序列，
避免回测热点循环里逐股票、逐截面重复调用。仅用 numpy，无 pandas 依赖。
"""
import numpy as np


def kline_to_arrays(kline: list[dict]) -> dict:
    """K 线序列转 numpy 数组。返回 {date, open, close, high, low, volume}。"""
    n = len(kline)
    close = np.empty(n, dtype=np.float64)
    high = np.empty(n, dtype=np.float64)
    low = np.empty(n, dtype=np.float64)
    volume = np.empty(n, dtype=np.float64)
    dates = [None] * n
    for i, row in enumerate(kline):
        dates[i] = row["date"]
        close[i] = row["close"]
        high[i] = row["high"]
        low[i] = row["low"]
        volume[i] = row["volume"]
    return {"date": dates, "close": close, "high": high, "low": low, "volume": volume}


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
    """N 日收益率标准差，前 n 项为 NaN。"""
    out = np.full_like(close, np.nan)
    if len(close) <= n:
        return out
    rets = _daily_returns(close)
    for i in range(n, len(close)):
        out[i] = np.std(rets[i - n + 1: i + 1], ddof=0)
    return out


def rsi_series(close: np.ndarray, n: int) -> np.ndarray:
    """RSI(n)，简化平均版（与 factors.py factor_rsi 一致），前 n 项为 NaN。"""
    out = np.full_like(close, np.nan)
    if len(close) <= n:
        return out
    diff = np.diff(close)
    gain = np.where(diff >= 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    for i in range(n, len(close)):
        avg_gain = np.mean(gain[i - n: i])
        avg_loss = np.mean(loss[i - n: i])
        if avg_loss == 0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def macd_series(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """MACD 柱值序列，前 slow+signal-1 项为 NaN。"""
    out = np.full_like(close, np.nan)
    if len(close) < slow + signal:
        return out
    ema_f = _ema(close, fast)
    ema_s = _ema(close, slow)
    dif = ema_f - ema_s
    valid_dif = dif[~np.isnan(dif)]
    if len(valid_dif) < signal:
        return out
    dea = _ema(valid_dif, signal)
    last_valid = valid_dif[-1]
    last_dea = dea[-1]
    out[-1] = last_valid - last_dea if not np.isnan(last_dea) else np.nan
    return out


def kdj_k_series(kline_arr: dict, n: int = 9, smooth: int = 3) -> np.ndarray:
    """KDJ-K 值序列，前 n-1 项为 NaN。"""
    close = kline_arr["close"]
    high = kline_arr["high"]
    low = kline_arr["low"]
    m = len(close)
    out = np.full(m, np.nan)
    if m < n:
        return out
    k_val = 50.0
    for idx in range(n - 1, m):
        hh = np.max(high[idx - n + 1: idx + 1])
        ll = np.min(low[idx - n + 1: idx + 1])
        if hh == ll:
            rsv = 50.0
        else:
            rsv = (close[idx] - ll) / (hh - ll) * 100.0
        k_val = (smooth - 1) / smooth * k_val + rsv / smooth
        out[idx] = k_val
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
    for i in range(n - 1, m):
        hh = np.max(high[i - n + 1: i + 1])
        ll = np.min(low[i - n + 1: i + 1])
        if hh == ll:
            out[i] = 0.0
        else:
            out[i] = (hh - close[i]) / (hh - ll) * 100.0
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
    for i in range(n - 1, m):
        window = tp[i - n + 1: i + 1]
        ma_tp = np.mean(window)
        md = np.mean(np.abs(window - ma_tp))
        if md == 0:
            out[i] = 0.0
        else:
            out[i] = (tp[i] - ma_tp) / (0.015 * md)
    return out


def boll_pct_series(kline_arr: dict, n: int = 20, k: float = 2.0) -> np.ndarray:
    """布林带 %B 序列，前 n-1 项为 NaN。"""
    close = kline_arr["close"]
    m = len(close)
    out = np.full(m, np.nan)
    if m < n:
        return out
    for i in range(n - 1, m):
        window = close[i - n + 1: i + 1]
        mu = np.mean(window)
        sd = np.std(window, ddof=0)
        upper = mu + k * sd
        lower = mu - k * sd
        if upper == lower:
            out[i] = 0.5
        else:
            out[i] = (close[i] - lower) / (upper - lower)
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
    for i in range(n, m):
        prev = close[i - n: i]
        hl = high[i - n + 1: i + 1] - low[i - n + 1: i + 1]
        amps = np.where(prev == 0, np.nan, hl / prev)
        if np.all(np.isnan(amps)):
            out[i] = np.nan
        else:
            out[i] = np.nanmean(amps)
    return out


def volume_ratio_series(kline_arr: dict, n: int = 5) -> np.ndarray:
    """量比序列，前 n 项为 NaN。"""
    vol = kline_arr["volume"]
    m = len(vol)
    out = np.full(m, np.nan)
    if m <= n:
        return out
    for i in range(n, m):
        base = np.mean(vol[i - n: i])
        out[i] = np.nan if base == 0 else vol[i] / base
    return out


def obv_trend_series(kline_arr: dict, n: int = 20) -> np.ndarray:
    """OBV 趋势序列，前 n 项为 NaN。"""
    close = kline_arr["close"]
    vol = kline_arr["volume"]
    m = len(close)
    out = np.full(m, np.nan)
    if m <= n:
        return out
    for i in range(n, m):
        diff = np.diff(close[i - n: i + 1])
        signs = np.where(diff > 0, 1.0, np.where(diff < 0, -1.0, 0.0))
        signed = np.sum(signs * vol[i - n + 1: i + 1])
        vol_sum = np.sum(vol[i - n + 1: i + 1])
        out[i] = np.nan if vol_sum == 0 else signed / vol_sum
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
    for i in range(n - 1, m):
        hh = np.max(high[i - n + 1: i + 1])
        ll = np.min(low[i - n + 1: i + 1])
        if hh == ll:
            out[i] = 0.5
        else:
            out[i] = (close[i] - ll) / (hh - ll)
    return out


def dist_high_series(close: np.ndarray, n: int = 240) -> np.ndarray:
    """距 N 日新高回撤序列。"""
    m = len(close)
    out = np.full(m, np.nan)
    for i in range(m):
        w = min(n, i + 1)
        if w < 20:
            continue
        hh = np.max(close[i - w + 1: i + 1])
        if hh == 0:
            out[i] = np.nan
        else:
            out[i] = close[i] / hh - 1.0
    return out


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
