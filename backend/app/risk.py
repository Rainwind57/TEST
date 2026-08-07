"""Barra 风格风险模型：多因子方差分解归因。

简化版（不依赖外部 Barra 数据）：
- 风格因子：动量、波动率、市值、估值（PE/PB 倒数）、换手率
- 对组合收益做因子回归，分解为 风格贡献 + 残差（特质风险）
- 输出各风格因子暴露、贡献、残差波动
"""
import numpy as np
from .factors import mean, std, zscore
from .numpy_factors import kline_to_arrays, compute_factor_series, series_at


# 风格因子：原 5 个 + size。beta/leverage 曾被加入，但：
# - beta 无基准行情，用自身收益回归恒等于 1.0（常数列，与截距完全共线）
# - leverage 依赖 debtRatio，而 fetch_quotes 不返回该字段，恒为 0.0
# 两者均产生退化常数列导致截面回归秩亏，故移除。
STYLE_FACTORS = ["momentum", "volatility", "turnover", "ep", "bp", "size"]
STYLE_LABELS = {
    "momentum": "动量", "volatility": "波动率", "turnover": "换手率",
    "ep": "盈利收益率", "bp": "账面市值比", "size": "市值(对数)",
}


def build_style_panel(stock_data: list[dict], n: int = 1) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Fama-MacBeth 时序面板：对每个历史截面回归得因子收益 beta_t。

    返回 (betas, factor_names, last_X, last_codes)：
    - betas: (n_periods, n_factors) 因子收益时序，用于估计因子收益协方差
    - last_X / last_codes: 最新截面暴露矩阵，用于组合归因
    turnover/ep/bp/size 无历史时序，用最新快照近似（已知简化）。
    """
    import math
    series = []
    for s in stock_data:
        kline = s.get("kline", [])
        quote = s.get("quote", {})
        if len(kline) < 30:
            continue
        arr = kline_to_arrays(kline)
        mom = compute_factor_series("momentum", arr)
        vol = compute_factor_series("volatility", arr)
        turnover = quote.get("turnover")
        pe = quote.get("pe")
        pb = quote.get("pb")
        mkt_cap = quote.get("mktCap")
        if any(v is None for v in [mom, vol, turnover, pe, pb]) or pe == 0 or pb == 0:
            continue
        s_size = math.log(max(mkt_cap, 1)) if mkt_cap else None
        series.append({"code": s["code"], "closes": arr["close"], "mom": mom, "vol": vol,
                       "turnover": turnover, "ep": 1.0 / pe, "bp": 1.0 / pb,
                       "size": s_size})
    if len(series) < 3:
        return np.zeros((0, 0)), STYLE_FACTORS, np.zeros((0, 0)), []

    ref = max(series, key=lambda s: len(s["closes"]))
    ref_len = len(ref["closes"])
    betas = []
    last_X = np.zeros((0, len(STYLE_FACTORS)))
    last_codes: list[str] = []
    X_t = last_X
    codes_t: list[str] = []
    for t in range(25, ref_len - n):
        rows, rets, codes_t = [], [], []
        for s in series:
            if t >= len(s["closes"]) or t + n >= len(s["closes"]):
                continue
            m = series_at(s["mom"], t)
            v = series_at(s["vol"], t)
            if m is None or v is None or s["closes"][t] == 0:
                continue
            row = [m, v, s["turnover"], s["ep"], s["bp"], s["size"] or 0.0]
            rows.append(row)
            rets.append(s["closes"][t + n] / s["closes"][t] - 1.0)
            codes_t.append(s["code"])
        if len(rows) < 10:
            continue
        X_t = _zscore_cols(np.array(rows, dtype=np.float64))
        betas.append(cross_section_regression(X_t, np.array(rets, dtype=np.float64)))
        last_X, last_codes = X_t, codes_t

    if not betas or last_X.size == 0:
        return np.zeros((0, 0)), STYLE_FACTORS, np.zeros((0, 0)), []
    return np.array(betas, dtype=np.float64), STYLE_FACTORS, last_X, last_codes


def _zscore_cols(X: np.ndarray) -> np.ndarray:
    """按列 z-score（截面标准化），常量列保留 0。"""
    out = np.zeros_like(X)
    for c in range(X.shape[1]):
        col = X[:, c]
        valid = col[~np.isnan(col)]
        if len(valid) < 2:
            continue
        m, s = valid.mean(), valid.std()
        if s != 0:
            out[:, c] = (col - m) / s
    return out


def factor_covariance(factor_returns_panel: np.ndarray, shrink: bool = True) -> np.ndarray:
    """因子收益时序协方差矩阵（Fama-MacBeth beta 时序估计）+ Ledoit-Wolf 收缩。

    输入应为 (n_periods, n_factors) 的因子收益时序。旧版误用截面暴露矩阵 np.cov(X)
    冒充因子收益协方差——对象完全错误（截面暴露共变 ≠ 因子收益时序共变）。
    shrink=True 时对齐噪声项做 Ledoit-Wolf 收缩，提升短面板（n_periods < n_factors×3）可靠性。
    """
    if factor_returns_panel.size == 0:
        return np.zeros((0, 0))
    if factor_returns_panel.ndim < 2 or factor_returns_panel.shape[0] < 2:
        k = factor_returns_panel.shape[-1] if factor_returns_panel.ndim >= 1 else 0
        return np.zeros((k, k))
    S = np.cov(factor_returns_panel, rowvar=False)
    if not shrink:
        return S
    n, k = factor_returns_panel.shape
    if n < k * 3:
        # 短面板：用 Ledoit-Wolf 收缩到对角
        d = np.diag(S).mean()
        target = d * np.eye(k)
        # 收缩强度 = k*(k+1) / (2*n*(k-1)) 粗略近似
        rho = min(1.0, k * (k + 1) / (2 * max(1, n) * max(1, k - 1)))
        return (1 - rho) * S + rho * target
    return S


def attribute_returns(weights: np.ndarray, X: np.ndarray, factor_returns: np.ndarray,
                      stock_returns: np.ndarray) -> dict:
    """收益归因：组合收益 = 风格因子贡献 + 残差。

    weights: 组合权重
    X: 风格暴露矩阵 (n_stocks, n_factors)
    factor_returns: 因子收益向量 (n_factors,)，可由 cross-section 回归得到
    stock_returns: 个股收益
    """
    if X.size == 0 or len(weights) != len(stock_returns):
        return {"exposures": [], "factorContribution": [], "totalFactor": 0.0,
                "residual": 0.0, "total": 0.0}

    port_exposure = weights @ X
    factor_contrib = port_exposure * factor_returns
    total_factor = float(np.sum(factor_contrib))
    total = float(weights @ stock_returns)
    residual = total - total_factor

    return {
        "exposures": port_exposure.tolist(),
        "factorContribution": factor_contrib.tolist(),
        "totalFactor": total_factor,
        "residual": residual,
        "total": total,
    }


def cross_section_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """截面回归求因子收益：y = X @ beta + epsilon，OLS 解。"""
    if X.size == 0 or len(y) != X.shape[0]:
        return np.zeros(X.shape[1] if X.ndim == 2 else 0)
    # 加截距项
    Xa = np.column_stack([np.ones(X.shape[0]), X])
    try:
        beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
        return beta[1:]  # 去掉截距
    except np.linalg.LinAlgError:
        return np.zeros(X.shape[1])


def risk_decomposition(weights: np.ndarray, X: np.ndarray, sigma_f: np.ndarray,
                       sigma_e: np.ndarray) -> dict:
    """风险分解：组合波动 = 系统性（因子）风险 + 特质风险。

    sigma_f: 因子协方差矩阵 (n_factors, n_factors)
    sigma_e: 个股特质方差向量 (n_stocks,)，用残差方差估计
    """
    if X.size == 0:
        return {"factorRisk": 0.0, "specificRisk": 0.0, "totalRisk": 0.0}
    port_exposure = weights @ X
    factor_var = float(port_exposure @ sigma_f @ port_exposure)
    specific_var = float(np.sum((weights ** 2) * sigma_e))
    total_var = factor_var + specific_var
    return {
        "factorRisk": float(np.sqrt(max(0, factor_var))),
        "specificRisk": float(np.sqrt(max(0, specific_var))),
        "totalRisk": float(np.sqrt(max(0, total_var))),
        "factorRiskPct": factor_var / total_var if total_var > 0 else 0.0,
    }


def estimate_specific_variances(stock_data: list[dict], X: np.ndarray, codes: list[str],
                                factor_returns: np.ndarray, n: int = 1) -> np.ndarray:
    """逐股特质方差估计（旧版路由层用截面残差方差标量广播到所有股票，过粗）。

    用每只股票的时序残差 r_t - X_t @ factor_returns 算其方差，输出 (n_stocks,)。
    样本不足（<20）时退化为截面均值，避免单股方差估计失真。
    """
    if X.size == 0 or not codes:
        return np.zeros(0)
    by_code = {s["code"]: s for s in stock_data}
    sigmas = np.zeros(len(codes))
    fallback_vals = []
    for i, code in enumerate(codes):
        s = by_code.get(code, {})
        kline = s.get("kline", [])
        if len(kline) < 30:
            continue
        arr = _safe_kline_arrays(kline)
        if arr is None:
            continue
        closes = arr["close"]
        rets = []
        xrows = []
        for t in range(25, len(closes) - n):
            row = _stock_style_row(s, arr, t)
            if row is None or closes[t] == 0:
                continue
            xrows.append(row)
            rets.append(closes[t + n] / closes[t] - 1.0)
        if len(xrows) < 20:
            continue
        Xt = _zscore_cols(np.array(xrows, dtype=np.float64))
        rt = np.array(rets, dtype=np.float64)
        resid = rt - Xt @ factor_returns
        sigmas[i] = float(np.var(resid, ddof=k + 1))
        fallback_vals.append(sigmas[i])
    mean_v = float(np.mean(fallback_vals)) if fallback_vals else 0.0
    sigmas = np.where(sigmas > 0, sigmas, mean_v)
    return sigmas


def _safe_kline_arrays(kline: list[dict]):
    """安全转换 K 线，失败返回 None（隔离单股异常不影响整组）。"""
    try:
        return kline_to_arrays(kline)
    except Exception:
        return None


def _stock_style_row(s: dict, arr: dict, t: int):
    """取某股 t 时刻的风格因子行（与 build_style_panel 截面一致）。"""
    import math
    mom = compute_factor_series("momentum", arr)
    vol = compute_factor_series("volatility", arr)
    m = series_at(mom, t)
    v = series_at(vol, t)
    quote = s.get("quote", {})
    turnover = quote.get("turnover")
    pe = quote.get("pe")
    pb = quote.get("pb")
    mkt_cap = quote.get("mktCap")
    if any(x is None for x in [m, v, turnover, pe, pb]) or pe == 0 or pb == 0:
        return None
    s_size = math.log(max(mkt_cap, 1)) if mkt_cap else 0.0
    return [m, v, turnover, 1.0 / pe, 1.0 / pb, s_size]


def value_at_risk(returns: np.ndarray, weights: np.ndarray | None = None,
                  alpha: float = 0.05) -> dict:
    """VaR（历史模拟法）：组合收益分布的 alpha 分位数。

    returns: (n_periods,) 组合收益时序，或 (n_periods, n_assets) 个股收益时序
    weights: 个股收益时需传权重，组合收益时为 None
    alpha: 置信水平尾部（0.05 = 95% VaR）
    """
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim == 2 and weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        port = arr @ w
    else:
        port = arr
    port = port[~np.isnan(port)]
    if len(port) < 30:
        return {"var": 0.0, "cvar": 0.0, "n": len(port), "warning": "样本不足，需≥30"}
    var = float(-np.quantile(port, alpha))
    cvar = float(-np.mean(port[port <= np.quantile(port, alpha)]))
    return {"var": var, "cvar": cvar, "n": len(port), "alpha": alpha}


def conditional_var(returns: np.ndarray, weights: np.ndarray | None = None,
                     alpha: float = 0.05) -> float:
    """CVaR/Expected Shortfall：尾部均值。便捷封装，仅返回数值。"""
    return value_at_risk(returns, weights, alpha)["cvar"]
