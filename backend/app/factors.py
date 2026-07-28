"""因子计算与统计函数（均为纯 Python 实现，逻辑与前端版本一致，已验证正确）。"""
import math


def mean(arr: list[float]) -> float:
    return sum(arr) / len(arr)


def std(arr: list[float]) -> float:
    m = mean(arr)
    return math.sqrt(mean([(v - m) ** 2 for v in arr]))


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    den = math.sqrt(dx2 * dy2)
    return 0.0 if den == 0 else num / den


def rank_of(arr: list[float]) -> list[float]:
    idx = sorted(range(len(arr)), key=lambda i: arr[i])
    r = [0.0] * len(arr)
    for rank_pos, original_idx in enumerate(idx):
        r[original_idx] = rank_pos + 1
    return r


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rank_of(xs), rank_of(ys))


def ols_regression(xs: list[float], ys: list[float]) -> dict:
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den = sum((x - mx) ** 2 for x in xs)
    b = 0.0 if den == 0 else num / den
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((ys[i] - (a + b * xs[i])) ** 2 for i in range(n))
    r2 = 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot
    return {"a": a, "b": b, "r2": r2, "n": n}


def ema_series(values: list[float], period: int) -> list:
    """指数移动平均序列，前 period-1 项为 None，第 period 项以简单均值作种子。"""
    if len(values) < period:
        return [None] * len(values)
    k = 2 / (period + 1)
    out = [None] * (period - 1)
    seed = mean(values[:period])
    out.append(seed)
    prev = seed
    for v in values[period:]:
        prev = v * k + prev * (1 - k)
        out.append(prev)
    return out


# ---------------- 因子公式（kline 为按日期升序的 [{date,open,close,high,low,volume}] 序列） ----------------

def _closes(kline):
    return [r["close"] for r in kline]


def _highs(kline):
    return [r["high"] for r in kline]


def _lows(kline):
    return [r["low"] for r in kline]


def _volumes(kline):
    return [r["volume"] for r in kline]


def factor_ma_dev(kline, i: int, n: int = 20):
    closes = _closes(kline)
    if i < n - 1:
        return None
    window = closes[i - n + 1: i + 1]
    ma = mean(window)
    return None if ma == 0 else (closes[i] - ma) / ma


def factor_momentum(kline, i: int, n: int = 20):
    closes = _closes(kline)
    if i < n:
        return None
    base = closes[i - n]
    return None if base == 0 else closes[i] / base - 1


def factor_volatility(kline, i: int, n: int = 20):
    closes = _closes(kline)
    if i < n:
        return None
    rets = []
    for k in range(i - n + 1, i + 1):
        prev = closes[k - 1]
        rets.append(0.0 if prev == 0 else closes[k] / prev - 1)
    return std(rets)


def factor_rsi(kline, i: int, n: int = 14):
    closes = _closes(kline)
    if i < n:
        return None
    gain = 0.0
    loss = 0.0
    for k in range(i - n + 1, i + 1):
        diff = closes[k] - closes[k - 1]
        if diff >= 0:
            gain += diff
        else:
            loss -= diff
    avg_gain, avg_loss = gain / n, loss / n
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def factor_macd(kline, i: int, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD 柱值（DIF - DEA）。"""
    if i < slow + signal:
        return None
    closes = _closes(kline)[: i + 1]
    ema_fast = ema_series(closes, fast)
    ema_slow = ema_series(closes, slow)
    dif = [None if (a is None or b is None) else a - b for a, b in zip(ema_fast, ema_slow)]
    valid_dif = [d for d in dif if d is not None]
    if len(valid_dif) < signal:
        return None
    dea = ema_series(valid_dif, signal)[-1]
    return None if dea is None else valid_dif[-1] - dea


def factor_kdj_k(kline, i: int, n: int = 9, smooth: int = 3):
    closes, highs, lows = _closes(kline), _highs(kline), _lows(kline)
    if i < n - 1:
        return None
    start = n - 1
    k_val = 50.0
    for idx in range(start, i + 1):
        hh, ll = max(highs[idx - n + 1: idx + 1]), min(lows[idx - n + 1: idx + 1])
        rsv = 50.0 if hh == ll else (closes[idx] - ll) / (hh - ll) * 100
        k_val = (smooth - 1) / smooth * k_val + rsv / smooth
    return k_val


def factor_wr(kline, i: int, n: int = 14):
    closes, highs, lows = _closes(kline), _highs(kline), _lows(kline)
    if i < n - 1:
        return None
    hh, ll = max(highs[i - n + 1: i + 1]), min(lows[i - n + 1: i + 1])
    return None if hh == ll else (hh - closes[i]) / (hh - ll) * 100


def factor_cci(kline, i: int, n: int = 14):
    closes, highs, lows = _closes(kline), _highs(kline), _lows(kline)
    if i < n - 1:
        return None
    tp = [(highs[k] + lows[k] + closes[k]) / 3 for k in range(i - n + 1, i + 1)]
    ma_tp = mean(tp)
    md = mean([abs(t - ma_tp) for t in tp])
    return 0.0 if md == 0 else (tp[-1] - ma_tp) / (0.015 * md)


def factor_boll_pct(kline, i: int, n: int = 20, k: float = 2.0):
    closes = _closes(kline)
    if i < n - 1:
        return None
    window = closes[i - n + 1: i + 1]
    m, s = mean(window), std(window)
    upper, lower = m + k * s, m - k * s
    return 0.5 if upper == lower else (closes[i] - lower) / (upper - lower)


def factor_amplitude(kline, i: int, n: int = 20):
    closes, highs, lows = _closes(kline), _highs(kline), _lows(kline)
    if i < n:
        return None
    amps = []
    for k in range(i - n + 1, i + 1):
        prev = closes[k - 1]
        if prev:
            amps.append((highs[k] - lows[k]) / prev)
    return mean(amps) if amps else None


def factor_volume_ratio(kline, i: int, n: int = 5):
    volumes = _volumes(kline)
    if i < n:
        return None
    base = mean(volumes[i - n: i])
    return None if base == 0 else volumes[i] / base


def factor_obv_trend(kline, i: int, n: int = 20):
    closes, volumes = _closes(kline), _volumes(kline)
    if i < n:
        return None
    signed, vol_sum = 0.0, 0.0
    for k in range(i - n + 1, i + 1):
        diff = closes[k] - closes[k - 1]
        sign = 1 if diff > 0 else (-1 if diff < 0 else 0)
        signed += sign * volumes[k]
        vol_sum += volumes[k]
    return None if vol_sum == 0 else signed / vol_sum


def factor_high_low_pos(kline, i: int, n: int = 20):
    closes, highs, lows = _closes(kline), _highs(kline), _lows(kline)
    if i < n - 1:
        return None
    hh, ll = max(highs[i - n + 1: i + 1]), min(lows[i - n + 1: i + 1])
    return 0.5 if hh == ll else (closes[i] - ll) / (hh - ll)


def factor_dist_high(kline, i: int, n: int = 240):
    closes = _closes(kline)
    n = min(n, i + 1)
    if n < 20:
        return None
    hh = max(closes[i - n + 1: i + 1])
    return None if hh == 0 else closes[i] / hh - 1


FACTORS = {
    "momentum5": {"label": "5日动量", "group": "technical", "direction": 1, "format": "pct", "calc": lambda k, i: factor_momentum(k, i, 5)},
    "momentum10": {"label": "10日动量", "group": "technical", "direction": 1, "format": "pct", "calc": lambda k, i: factor_momentum(k, i, 10)},
    "momentum": {"label": "20日动量", "group": "technical", "direction": 1, "format": "pct", "calc": lambda k, i: factor_momentum(k, i, 20)},
    "momentum60": {"label": "60日动量", "group": "technical", "direction": 1, "format": "pct", "calc": lambda k, i: factor_momentum(k, i, 60)},
    "momentum120": {"label": "120日动量", "group": "technical", "direction": 1, "format": "pct", "calc": lambda k, i: factor_momentum(k, i, 120)},
    "ma_dev": {"label": "MA20偏离度", "group": "technical", "direction": 1, "format": "pct", "calc": lambda k, i: factor_ma_dev(k, i, 20)},
    "ma5_dev": {"label": "MA5偏离度", "group": "technical", "direction": 1, "format": "pct", "calc": lambda k, i: factor_ma_dev(k, i, 5)},
    "ma60_dev": {"label": "MA60偏离度", "group": "technical", "direction": 1, "format": "pct", "calc": lambda k, i: factor_ma_dev(k, i, 60)},
    "volatility": {"label": "20日波动率", "group": "technical", "direction": -1, "format": "pct", "calc": lambda k, i: factor_volatility(k, i, 20)},
    "volatility60": {"label": "60日波动率", "group": "technical", "direction": -1, "format": "pct", "calc": lambda k, i: factor_volatility(k, i, 60)},
    "rsi": {"label": "RSI(14)", "group": "technical", "direction": 1, "format": "num", "calc": lambda k, i: factor_rsi(k, i, 14)},
    "rsi6": {"label": "RSI(6)", "group": "technical", "direction": 1, "format": "num", "calc": lambda k, i: factor_rsi(k, i, 6)},
    "macd": {"label": "MACD柱", "group": "technical", "direction": 1, "format": "num", "calc": lambda k, i: factor_macd(k, i)},
    "kdj_k": {"label": "KDJ-K值", "group": "technical", "direction": 1, "format": "num", "calc": lambda k, i: factor_kdj_k(k, i)},
    "wr14": {"label": "威廉指标WR(14)", "group": "technical", "direction": -1, "format": "num", "calc": lambda k, i: factor_wr(k, i, 14)},
    "cci14": {"label": "CCI(14)", "group": "technical", "direction": 1, "format": "num", "calc": lambda k, i: factor_cci(k, i, 14)},
    "boll_pct": {"label": "布林带%B", "group": "technical", "direction": 1, "format": "pct", "calc": lambda k, i: factor_boll_pct(k, i)},
    "amplitude20": {"label": "20日平均振幅", "group": "technical", "direction": -1, "format": "pct", "calc": lambda k, i: factor_amplitude(k, i, 20)},
    "volume_ratio5": {"label": "量比(5日)", "group": "technical", "direction": 1, "format": "num", "calc": lambda k, i: factor_volume_ratio(k, i, 5)},
    "obv_trend": {"label": "OBV趋势(20日)", "group": "technical", "direction": 1, "format": "pct", "calc": lambda k, i: factor_obv_trend(k, i, 20)},
    "high_low_pos": {"label": "20日高低区间位置", "group": "technical", "direction": 1, "format": "pct", "calc": lambda k, i: factor_high_low_pos(k, i, 20)},
    "dist_52w_high": {"label": "距52周新高回撤", "group": "technical", "direction": 1, "format": "pct", "calc": lambda k, i: factor_dist_high(k, i, 240)},
}


# ---------------- 快照类因子（来自新浪全市场行情快照，无需拉取K线，选股效率更高） ----------------

SNAPSHOT_FACTORS = {
    "pct_chg": {"label": "当日涨跌幅", "group": "quant", "direction": 1, "format": "pct_raw"},
    "turnover": {"label": "换手率", "group": "quant", "direction": 1, "format": "pct_raw"},
    "amount": {"label": "成交额(亿)", "group": "quant", "direction": 1, "format": "num"},
    "pe": {"label": "市盈率(动态)", "group": "fundamental", "direction": -1, "format": "num"},
    "pb": {"label": "市净率", "group": "fundamental", "direction": -1, "format": "num"},
    "ep": {"label": "盈利收益率(1/PE)", "group": "fundamental", "direction": 1, "format": "pct"},
    "bp": {"label": "账面市值比(1/PB)", "group": "fundamental", "direction": 1, "format": "pct"},
    "mkt_cap": {"label": "总市值(亿)", "group": "fundamental", "direction": -1, "format": "num"},
    "circ_mkt_cap": {"label": "流通市值(亿)", "group": "fundamental", "direction": -1, "format": "num"},
}

_SNAPSHOT_FIELD_MAP = {
    "pct_chg": "pctChg", "turnover": "turnover", "mkt_cap": "mktCap", "circ_mkt_cap": "circMktCap",
    "pe": "pe", "pb": "pb",
}


def snapshot_factor_value(row: dict, key: str):
    if key == "amount":
        amt = row.get("amount")
        return None if amt is None else amt / 1e8
    if key == "ep":
        pe = row.get("pe")
        return None if not pe else 1 / pe
    if key == "bp":
        pb = row.get("pb")
        return None if not pb else 1 / pb
    field = _SNAPSHOT_FIELD_MAP.get(key)
    return row.get(field) if field else None


# ---------------- 多因子打分 / 分层回测辅助函数 ----------------

def zscore(values: list) -> list[float]:
    valid = [v for v in values if v is not None]
    if len(valid) < 2:
        return [0.0 for _ in values]
    m = mean(valid)
    s = std(valid)
    if s == 0:
        return [0.0 for _ in values]
    return [0.0 if v is None else (v - m) / s for v in values]


def composite_score(rows: list[dict], specs: list[dict]) -> list[dict]:
    """specs: [{key, weight, direction}]，对 rows 做加权 z-score 打分（自动忽略缺失因子）。"""
    columns = {spec["key"]: zscore([r.get(spec["key"]) for r in rows]) for spec in specs}

    out = []
    for idx, row in enumerate(rows):
        detail = {}
        total, weight_sum = 0.0, 0.0
        for spec in specs:
            key = spec["key"]
            weight = spec.get("weight", 1.0)
            direction = spec.get("direction", 1)
            raw = row.get(key)
            z = columns[key][idx] * direction
            detail[key] = {"raw": raw, "z": z}
            if raw is not None:
                total += z * weight
                weight_sum += abs(weight)
        score = total / weight_sum if weight_sum > 0 else 0.0
        out.append({**row, "factorDetail": detail, "score": score})
    return out


def bucket_index(value: float, sorted_values: list[float], groups: int) -> int:
    """确定 value 在 sorted_values 分布中的分位组编号（0 = 最低组）。"""
    n = len(sorted_values)
    if n == 0:
        return 0
    pos = sum(1 for v in sorted_values if v <= value)
    idx = int(pos * groups / n)
    return min(idx, groups - 1)
