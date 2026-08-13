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


def _weighted_linear_fit(xs: list[float], ys: list[float], w: list[float]) -> tuple:
    sw = sum(w)
    wmx = sum(w[i] * xs[i] for i in range(len(xs))) / sw
    wmy = sum(w[i] * ys[i] for i in range(len(xs))) / sw
    num = sum(w[i] * (xs[i] - wmx) * (ys[i] - wmy) for i in range(len(xs)))
    den = sum(w[i] * (xs[i] - wmx) ** 2 for i in range(len(xs)))
    b = 0.0 if den == 0 else num / den
    a = wmy - b * wmx
    return a, b


def _linear_r2(xs: list[float], ys: list[float], a: float, b: float) -> float:
    my = mean(ys)
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((ys[i] - (a + b * xs[i])) ** 2 for i in range(len(xs)))
    return 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot


def ridge_regression(xs: list[float], ys: list[float], alpha: float = 1.0) -> dict:
    """岭回归（L2 正则，截距不惩罚）：闭式解，对异常斜率有收缩效果。"""
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    xc = [x - mx for x in xs]
    yc = [y - my for y in ys]
    den = sum(x * x for x in xc) + alpha
    b = 0.0 if den == 0 else sum(xc[i] * yc[i] for i in range(n)) / den
    a = my - b * mx
    return {"a": a, "b": b, "r2": _linear_r2(xs, ys, a, b), "n": n}


def huber_regression(xs: list[float], ys: list[float], delta: float = 1.345, iters: int = 25) -> dict:
    """稳健回归（Huber M-estimator，IRLS 迭代加权最小二乘），降低异常样本对拟合的影响。"""
    n = len(xs)
    base = ols_regression(xs, ys)
    a, b = base["a"], base["b"]
    for _ in range(iters):
        resid = [ys[i] - (a + b * xs[i]) for i in range(n)]
        mad = mean([abs(r) for r in resid])
        scale = mad / 0.6745 if mad else 1e-9
        weights = [1.0 if abs(r) / scale <= delta else delta * scale / max(abs(r), 1e-12) for r in resid]
        new_a, new_b = _weighted_linear_fit(xs, ys, weights)
        if abs(new_a - a) < 1e-9 and abs(new_b - b) < 1e-9:
            a, b = new_a, new_b
            break
        a, b = new_a, new_b
    return {"a": a, "b": b, "r2": _linear_r2(xs, ys, a, b), "n": n}


def quantile_regression(xs: list[float], ys: list[float], tau: float = 0.5, iters: int = 30) -> dict:
    """分位数回归（IRLS 近似，tau=0.5 即中位数/LAD 回归），对极端收益样本更不敏感。"""
    n = len(xs)
    base = ols_regression(xs, ys)
    a, b = base["a"], base["b"]
    eps = 1e-6
    for _ in range(iters):
        resid = [ys[i] - (a + b * xs[i]) for i in range(n)]
        weights = [(tau if r >= 0 else (1 - tau)) / max(abs(r), eps) for r in resid]
        new_a, new_b = _weighted_linear_fit(xs, ys, weights)
        if abs(new_a - a) < 1e-9 and abs(new_b - b) < 1e-9:
            a, b = new_a, new_b
            break
        a, b = new_a, new_b
    return {"a": a, "b": b, "r2": _linear_r2(xs, ys, a, b), "n": n}


def _solve_linear_system(a: list[list[float]], b: list[float]) -> list[float]:
    """高斯消元（列主元）求解 Ax=b，a 为 n x n 矩阵。"""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            continue
        m[col], m[pivot] = m[pivot], m[col]
        pv = m[col][col]
        m[col] = [v / pv for v in m[col]]
        for r in range(n):
            if r != col and m[r][col]:
                factor = m[r][col]
                m[r] = [m[r][k] - factor * m[col][k] for k in range(n + 1)]
    return [m[i][n] for i in range(n)]


def multi_ols(rows: list[list[float]], ys: list[float]) -> dict:
    """多元线性回归（含截距），rows[i] 为第 i 个样本各因子取值。返回 intercept/coefs/r2。"""
    n = len(ys)
    p = len(rows[0]) if rows else 0
    x = [[1.0] + list(r) for r in rows]
    k = p + 1
    xtx = [[sum(x[i][c1] * x[i][c2] for i in range(n)) for c2 in range(k)] for c1 in range(k)]
    xty = [sum(x[i][c1] * ys[i] for i in range(n)) for c1 in range(k)]
    beta = _solve_linear_system(xtx, xty)
    my = mean(ys)
    ss_tot = sum((y - my) ** 2 for y in ys)
    preds = [sum(beta[c] * x[i][c] for c in range(k)) for i in range(n)]
    ss_res = sum((ys[i] - preds[i]) ** 2 for i in range(n))
    r2 = 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot
    return {"intercept": beta[0], "coefs": beta[1:], "r2": r2, "n": n}


def poly_regression(xs: list[float], ys: list[float], degree: int = 2) -> dict:
    """多项式回归（默认二次），返回按幂次排列的系数 coefs=[c0,c1,c2,...]。"""
    rows = [[x ** d for d in range(1, degree + 1)] for x in xs]
    fit = multi_ols(rows, ys)
    return {"coefs": [fit["intercept"]] + fit["coefs"], "r2": fit["r2"], "n": fit["n"]}


def poly_predict(coefs: list[float], x: float) -> float:
    return sum(c * x ** i for i, c in enumerate(coefs))


REGRESSION_METHODS = {
    "ols": {"label": "普通最小二乘(OLS)", "fn": lambda xs, ys: ols_regression(xs, ys)},
    "ridge": {"label": "岭回归(Ridge)", "fn": lambda xs, ys: ridge_regression(xs, ys, 1.0)},
    "huber": {"label": "稳健回归(Huber)", "fn": lambda xs, ys: huber_regression(xs, ys)},
    "quantile": {"label": "分位数回归(中位数/LAD)", "fn": lambda xs, ys: quantile_regression(xs, ys, 0.5)},
    "poly2": {"label": "二次多项式回归", "fn": lambda xs, ys: poly_regression(xs, ys, 2)},
}


def fit_regression(method: str, xs: list[float], ys: list[float]) -> dict:
    """统一出参：{coefs:[c0,c1,...], r2, n}，coefs 按幂次排列（c0 为截距）。"""
    meta = REGRESSION_METHODS.get(method, REGRESSION_METHODS["ols"])
    result = meta["fn"](xs, ys)
    if "coefs" in result:
        return result
    return {"coefs": [result["a"], result["b"]], "r2": result["r2"], "n": result["n"]}


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


def factor_dist_low(kline, i: int, n: int = 240):
    """距 N 日低点涨幅：close / 窗口内最低 - 1（窗口随短历史收缩）。"""
    closes = _closes(kline)
    n = min(n, i + 1)
    if n < 20:
        return None
    ll = min(closes[i - n + 1: i + 1])
    return None if ll == 0 else closes[i] / ll - 1


def factor_skew(kline, i: int, n: int = 20):
    """N 日收益率偏度（>0 右尾风险，方向 -1）。"""
    closes = _closes(kline)
    if i < n:
        return None
    rets = []
    for k in range(i - n + 1, i + 1):
        prev = closes[k - 1]
        rets.append(0.0 if prev == 0 else closes[k] / prev - 1)
    s = std(rets)
    if s == 0:
        return 0.0
    m = mean(rets)
    return mean([(r - m) ** 3 for r in rets]) / s ** 3


def factor_kurt(kline, i: int, n: int = 20):
    """N 日收益率超额峰度（>0 肥尾，方向 -1）。"""
    closes = _closes(kline)
    if i < n:
        return None
    rets = []
    for k in range(i - n + 1, i + 1):
        prev = closes[k - 1]
        rets.append(0.0 if prev == 0 else closes[k] / prev - 1)
    s = std(rets)
    if s == 0:
        return 0.0
    m = mean(rets)
    return mean([(r - m) ** 4 for r in rets]) / s ** 4 - 3.0


def factor_down_vol(kline, i: int, n: int = 20):
    """N 日下行波动率：只统计负收益的标准差。"""
    closes = _closes(kline)
    if i < n:
        return None
    neg = []
    for k in range(i - n + 1, i + 1):
        prev = closes[k - 1]
        r = 0.0 if prev == 0 else closes[k] / prev - 1
        if r < 0:
            neg.append(r)
    return std(neg) if neg else 0.0


def factor_max_drawdown(kline, i: int, n: int = 60):
    """N 日最大回撤（负值，越深风险越大，方向 -1）。"""
    closes = _closes(kline)
    if i < n - 1:
        return None
    window = closes[i - n + 1: i + 1]
    peak, mdd = window[0], 0.0
    for c in window:
        if c > peak:
            peak = c
        dd = c / peak - 1.0 if peak else 0.0
        if dd < mdd:
            mdd = dd
    return mdd


def factor_ma_align(kline, i: int, n1: int = 5, n2: int = 20, n3: int = 60):
    """均线排列强度：(MA5-MA20)/MA20 × (MA20-MA60)/MA60，多头排列为正。"""
    closes = _closes(kline)
    if i < n3 - 1:
        return None
    ma5 = mean(closes[i - n1 + 1: i + 1])
    ma20 = mean(closes[i - n2 + 1: i + 1])
    ma60 = mean(closes[i - n3 + 1: i + 1])
    if ma20 == 0 or ma60 == 0:
        return None
    return (ma5 - ma20) / ma20 * (ma20 - ma60) / ma60


def factor_ema_slope(kline, i: int, n: int = 20):
    """EMA(n) 斜率：(EMA_t/EMA_{t-1})-1。"""
    closes = _closes(kline)
    if i < n:
        return None
    e = ema_series(closes[: i + 1], n)
    cur, prev = e[-1], e[-2]
    if cur is None or prev is None or prev == 0:
        return None
    return cur / prev - 1


def factor_bias(kline, i: int, n: int = 60):
    """BIAS(n) 乖离率：(close-MA)/MA。"""
    closes = _closes(kline)
    if i < n - 1:
        return None
    ma = mean(closes[i - n + 1: i + 1])
    return None if ma == 0 else closes[i] / ma - 1


def factor_donchian_break(kline, i: int, n: int = 20):
    """唐奇安通道突破：close / 前 n 日最高 - 1（含当日，>0 突破）。"""
    closes = _closes(kline)
    if i < n - 1:
        return None
    hh = max(closes[i - n + 1: i + 1])
    return None if hh == 0 else closes[i] / hh - 1


def factor_vol_corr(kline, i: int, n: int = 20):
    """N 日量价相关：收盘价与成交量的 Pearson 相关。"""
    closes, volumes = _closes(kline), _volumes(kline)
    if i < n:
        return None
    return pearson(closes[i - n: i], volumes[i - n: i])


def factor_vol_change(kline, i: int, n: int = 5):
    """量能突增：今日量 / N 日均量 - 1。"""
    volumes = _volumes(kline)
    if i < n:
        return None
    base = mean(volumes[i - n: i])
    return None if base == 0 else volumes[i] / base - 1


def factor_mom_accel(kline, i: int):
    """动量加速度：20 日动量 - 10 日动量（短期动量增强/减弱）。"""
    m20 = factor_momentum(kline, i, 20)
    m10 = factor_momentum(kline, i, 10)
    return None if m20 is None or m10 is None else m20 - m10


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
    "momentum30": {"label": "30日动量", "group": "technical", "direction": 1, "format": "pct", "calc": lambda k, i: factor_momentum(k, i, 30)},
    "momentum90": {"label": "90日动量", "group": "technical", "direction": 1, "format": "pct", "calc": lambda k, i: factor_momentum(k, i, 90)},
    "mom_accel": {"label": "动量加速度(20-10)", "group": "technical", "direction": 1, "format": "pct", "calc": lambda k, i: factor_mom_accel(k, i)},
    "ma_align": {"label": "均线排列强度", "group": "technical", "direction": 1, "format": "num", "calc": lambda k, i: factor_ma_align(k, i)},
    "ema_slope20": {"label": "EMA20斜率", "group": "technical", "direction": 1, "format": "pct", "calc": lambda k, i: factor_ema_slope(k, i, 20)},
    "bias60": {"label": "BIAS60乖离率", "group": "technical", "direction": 1, "format": "pct", "calc": lambda k, i: factor_bias(k, i, 60)},
    "donchian_break20": {"label": "唐奇安20日突破", "group": "technical", "direction": 1, "format": "pct", "calc": lambda k, i: factor_donchian_break(k, i, 20)},
    "dist_52w_low": {"label": "距52周低点涨幅", "group": "technical", "direction": 1, "format": "pct", "calc": lambda k, i: factor_dist_low(k, i, 240)},
    "skew20": {"label": "20日收益偏度", "group": "technical", "direction": -1, "format": "num", "calc": lambda k, i: factor_skew(k, i, 20)},
    "kurt20": {"label": "20日收益峰度", "group": "technical", "direction": -1, "format": "num", "calc": lambda k, i: factor_kurt(k, i, 20)},
    "down_vol20": {"label": "20日下行波动率", "group": "technical", "direction": -1, "format": "pct", "calc": lambda k, i: factor_down_vol(k, i, 20)},
    "max_drawdown60": {"label": "60日最大回撤", "group": "technical", "direction": -1, "format": "pct", "calc": lambda k, i: factor_max_drawdown(k, i, 60)},
    "vol_corr20": {"label": "20日量价相关", "group": "technical", "direction": 1, "format": "num", "calc": lambda k, i: factor_vol_corr(k, i, 20)},
    "vol_change5": {"label": "量能突增(5日)", "group": "technical", "direction": 1, "format": "pct", "calc": lambda k, i: factor_vol_change(k, i, 5)},
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
    "roe": {"label": "ROE(%)", "group": "fundamental", "direction": 1, "format": "num"},
    "net_margin": {"label": "净利率(%)", "group": "fundamental", "direction": 1, "format": "num"},
    "revenue_yoy": {"label": "营收同比(%)", "group": "fundamental", "direction": 1, "format": "num"},
    "profit_yoy": {"label": "净利同比(%)", "group": "fundamental", "direction": 1, "format": "num"},
    "main_net_pct": {"label": "主力净流入占比", "group": "moneyflow", "direction": 1, "format": "pct"},
    # 以下四个：adapters.fetch_finance_summary 已拉取但旧版未注册进 SNAPSHOT_FACTORS（数据白拉）
    "gross_margin": {"label": "毛利率(%)", "group": "fundamental", "direction": 1, "format": "num"},
    "debt_ratio": {"label": "资产负债率(%)", "group": "fundamental", "direction": -1, "format": "num"},
    "eps": {"label": "每股收益", "group": "fundamental", "direction": 1, "format": "num"},
    "bps": {"label": "每股净资产", "group": "fundamental", "direction": 1, "format": "num"},
    # 质量类：ROA（新增，由总资产/净利润算）；北向资金类：个股持股比例（新增数据源）
    "roa": {"label": "ROA(%)", "group": "fundamental", "direction": 1, "format": "num"},
    "north_holding_pct": {"label": "北向持股比例(%)", "group": "moneyflow", "direction": 1, "format": "num"},
    # 行业因子（需通过 adapters.fetch_sector_map 拉取）
    "sector": {"label": "申万一级行业", "group": "sector", "direction": 0, "format": "categorical"},
}

# 行业因子（中性化用，非打分因子，不参与 composite_score 直接排序）
SECTOR_FACTORS = {
    "sector": {"label": "申万一级行业", "group": "sector"},
}

# 宏观因子（需通过 adapters.fetch_macro_indicator 拉取）
MACRO_FACTORS = {
    "cpi": {"label": "CPI当月同比(%)", "group": "macro", "direction": 0, "format": "pct"},
    "ppi": {"label": "PPI当月同比(%)", "group": "macro", "direction": 0, "format": "pct"},
    "pmi": {"label": "制造业PMI", "group": "macro", "direction": 0, "format": "num"},
    "m2": {"label": "M2同比(%)", "group": "macro", "direction": 0, "format": "pct"},
}

_SNAPSHOT_FIELD_MAP = {
    "pct_chg": "pctChg", "turnover": "turnover", "mkt_cap": "mktCap", "circ_mkt_cap": "circMktCap",
    "pe": "pe", "pb": "pb",
    # 资金流/财务快照因子（由 selection.run_select 勾选时批量拉取填充 row，非行情快照自带）
    "roe": "roe", "net_margin": "net_margin", "revenue_yoy": "revenue_yoy",
    "profit_yoy": "profit_yoy", "main_net_pct": "main_net_pct",
    "gross_margin": "gross_margin", "debt_ratio": "debt_ratio",
    "eps": "eps", "bps": "bps",
    "roa": "roa", "north_holding_pct": "north_holding_pct",
    "sector": "sector",
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
    if key == "sector":
        return row.get("sector", "")
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


def neutralized_zscore(values: list, exposures: list[list[float]]) -> list[float]:
    """行业/市值横截面中性化：对 values 关于 exposures 做多元 OLS 取残差，再 z-score。

    exposures[i] 为第 i 个样本的风格暴露向量（如 [log市值, 行业哑变量...]）。
    残差即剥离风格暴露后的纯因子值。与 zscore 同返回对齐列表，None 视为缺失跳过。
    """
    paired = [(v, e) for v, e in zip(values, exposures) if v is not None]
    if len(paired) < 2:
        return [0.0 for _ in values]
    ys = [p[0] for p in paired]
    xs_rows = [p[1] for p in paired]
    if not any(any(x != 0 for x in row) for row in xs_rows):
        return zscore(values)
    fit = multi_ols(xs_rows, ys)
    coefs = fit["coefs"]
    intercept = fit["intercept"]
    valid_idx = [i for i, v in enumerate(values) if v is not None]
    residuals = []
    for k, i in enumerate(valid_idx):
        pred = intercept + sum(coefs[c] * xs_rows[k][c] for c in range(len(coefs)))
        residuals.append(ys[k] - pred)
    z = zscore(residuals)
    out = [0.0] * len(values)
    for k, i in enumerate(valid_idx):
        out[i] = z[k]
    return out


def sector_dummies(codes: list[str], sector_map: dict[str, str]) -> list[list[float]]:
    """将行业映射转为哑变量矩阵（一维行业，每列一个行业，每行只有一个1）。

    用于 neutralized_zscore 的 exposures 参数，剥离行业影响。
    """
    unique_sectors = sorted(set(s for s in sector_map.values() if s))
    if not unique_sectors:
        return [[0.0] for _ in codes]
    idx = {s: i for i, s in enumerate(unique_sectors)}
    dummies = []
    for code in codes:
        s = sector_map.get(code, "")
        row = [0.0] * len(unique_sectors)
        if s in idx:
            row[idx[s]] = 1.0
        dummies.append(row)
    return dummies


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
    """确定 value 在 sorted_values 分布中的分位组编号（0 = 最低组）。并列值分布均匀。"""
    n = len(sorted_values)
    if n == 0:
        return 0
    pos = sum(1 for v in sorted_values if v < value)
    ties = sum(1 for v in sorted_values if v == value)
    if ties > 1:
        avg_pos = pos + (ties - 1) / 2.0
    else:
        avg_pos = float(pos)
    idx = int(avg_pos * groups / max(n, 1))
    return min(idx, groups - 1)


# ---------------- 绩效指标 ----------------

TRADING_DAYS = 252


def annualized_return(period_returns: list[float], periods_per_year: float = TRADING_DAYS) -> float:
    if not period_returns:
        return 0.0
    cum = 1.0
    for r in period_returns:
        cum *= (1.0 + r)
    years = len(period_returns) / periods_per_year
    if years <= 0:
        return 0.0
    return cum ** (1.0 / years) - 1.0


def annualized_volatility(period_returns: list[float], periods_per_year: float = TRADING_DAYS) -> float:
    if len(period_returns) < 2:
        return 0.0
    return std(period_returns) * math.sqrt(periods_per_year)


def sharpe_ratio(period_returns: list[float], rf_annual: float = 0.0, periods_per_year: float = TRADING_DAYS) -> float:
    if not period_returns:
        return 0.0
    rf_period = rf_annual / periods_per_year
    excess = [r - rf_period for r in period_returns]
    s = std(excess)
    if s == 0:
        return 0.0
    return mean(excess) / s * math.sqrt(periods_per_year)


def sortino_ratio(period_returns: list[float], rf_annual: float = 0.0, periods_per_year: float = TRADING_DAYS) -> float:
    if not period_returns:
        return 0.0
    rf_period = rf_annual / periods_per_year
    excess = [r - rf_period for r in period_returns]
    downside = [e for e in excess if e < 0]
    if not downside:
        return 0.0
    dd_std = math.sqrt(mean([e * e for e in downside]))
    if dd_std == 0:
        return 0.0
    return mean(excess) / dd_std * math.sqrt(periods_per_year)


def max_drawdown(equity_curve: list[float]) -> float:
    """返回最大回撤（负值）。equity_curve 为累计净值序列（起点 1.0）。"""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    mdd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (v - peak) / peak
            if dd < mdd:
                mdd = dd
    return mdd


def calmar_ratio(period_returns: list[float], periods_per_year: float = TRADING_DAYS) -> float:
    ann = annualized_return(period_returns, periods_per_year)
    eq = [1.0]
    for r in period_returns:
        eq.append(eq[-1] * (1.0 + r))
    mdd = max_drawdown(eq)
    if mdd == 0:
        return 0.0
    return ann / abs(mdd)


def win_rate(period_returns: list[float]) -> float:
    if not period_returns:
        return 0.0
    return sum(1 for r in period_returns if r > 0) / len(period_returns)


def information_coefficient_stats(ic_series: list[float], periods_per_year: float = TRADING_DAYS) -> dict:
    """IC 统计。ICIR = meanIC/stdIC × √(年化采样数)；
    periods_per_year 应传 252/n（n=调仓间隔），旧版误用 √len(总期数) 导致口径错误。
    附加 t 统计量（mean / (std/√n)）与双侧 p 值（正态近似）。"""
    if not ic_series:
        return {"meanIc": 0.0, "icIr": 0.0, "icWinRate": 0.0, "tStat": 0.0, "pValue": 1.0}
    n = len(ic_series)
    m = mean(ic_series)
    s = std(ic_series)
    ir = 0.0 if s == 0 else m / s * math.sqrt(periods_per_year)
    if n < 2 or s == 0:
        t = 0.0
        p = 1.0 if s == 0 else 0.0
    else:
        t = m / (s / math.sqrt(n))
        # 正态近似双尾 p 值（自由度较大时与 t 分布几乎一致）
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))
    return {"meanIc": m, "icIr": ir, "icWinRate": sum(1 for v in ic_series if v > 0) / n,
            "tStat": t, "pValue": p}


# ---------------- 交易成本模型 ----------------

def round_trip_cost_rate(commission_rate: float, stamp_duty: float, slippage: float) -> float:
    """单次调仓的往返成本率（买卖双边）：佣金双边 + 印花税卖单 + 滑点双边。"""
    return 2.0 * commission_rate + stamp_duty + 2.0 * slippage


# ---------------- 用户自定义因子（组合式）解析 ----------------

def compute_user_factor_scores(rows: list[dict], definition: dict) -> list[float]:
    """组合式自定义因子：对 definition.factors（[{key,weight,direction}]）做加权 z-score，
    返回与 rows 对齐的得分列表。"""
    specs = definition.get("factors") or []
    if not specs:
        return [0.0 for _ in rows]
    columns = {spec["key"]: zscore([r.get(spec["key"]) for r in rows]) for spec in specs}
    scores = []
    for idx, row in enumerate(rows):
        total, wsum = 0.0, 0.0
        for spec in specs:
            key = spec["key"]
            weight = spec.get("weight", 1.0)
            direction = spec.get("direction", 1)
            raw = row.get(key)
            if raw is None:
                continue
            total += columns[key][idx] * direction * weight
            wsum += abs(weight)
        scores.append(total / wsum if wsum > 0 else 0.0)
    return scores
