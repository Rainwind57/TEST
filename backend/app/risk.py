"""Barra 风格风险模型：多因子方差分解归因。

- 风格因子：动量、波动率、市值、估值（PE/PB 倒数）、换手率
- 行业因子：申万行业哑变量（动态从行情源拉取）
- 对组合收益做因子回归，分解为 风格贡献 + 行业贡献 + 残差（特质风险）
- 输出各风格因子暴露、贡献、残差波动

注意：turnover/ep/bp/size 仍用当日快照（数据源不提供历史时序），
前视风险已在 snapshotWarning 中明示。行业因子使用最新分类，同样属快照。
"""
import numpy as np
from collections import defaultdict
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

# 行业因子：动态填充，构建面板时根据 sector_map 确定
INDUSTRY_FACTORS: list[str] = []
INDUSTRY_LABELS: dict[str, str] = {}


async def _build_industry_factors(codes: list[str]) -> dict[str, list[float]]:
    """拉取行业映射并构建行业哑变量列定义。

    返回 {code: [dummy_values...]}；INDUSTRY_FACTORS / INDUSTRY_LABELS 被填充。
    失败时返回空 dict，行业因子被跳过（退化为纯风格模型）。
    """
    global INDUSTRY_FACTORS, INDUSTRY_LABELS
    try:
        from . import adapters
        sector_map = await adapters.fetch_sector_map()
        if not sector_map:
            return {}
        unique = sorted(set(s for s in sector_map.values() if s))
        if not unique:
            return {}
        INDUSTRY_FACTORS = [f"industry_{i}" for i in range(len(unique))]
        INDUSTRY_LABELS = {f"industry_{i}": s for i, s in enumerate(unique)}
        idx = {s: i for i, s in enumerate(unique)}
        out = {}
        for code in codes:
            s = sector_map.get(code, "")
            row = [0.0] * len(unique)
            if s in idx:
                row[idx[s]] = 1.0
            out[code] = row
        return out
    except Exception:
        INDUSTRY_FACTORS = []
        INDUSTRY_LABELS = {}
        return {}


def all_factor_names() -> list[str]:
    """当前完整因子名列表：风格 + 行业。"""
    return STYLE_FACTORS + INDUSTRY_FACTORS


def all_factor_labels() -> dict[str, str]:
    """当前完整因子 label 映射。"""
    out = dict(STYLE_LABELS)
    out.update(INDUSTRY_LABELS)
    return out


async def build_style_panel(stock_data: list[dict], n: int = 1) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Fama-MacBeth 时序面板：对每个历史截面回归得因子收益 beta_t。

    返回 (betas, factor_names, last_X, last_codes)：
    - betas: (n_periods, n_factors) 因子收益时序，用于估计因子收益协方差
    - last_X / last_codes: 最新截面暴露矩阵，用于组合归因
    turnover/ep/bp/size 无历史时序，用最新快照近似（已知简化）。
    行业因子使用最新分类映射到所有历史截面（快照）。
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
        return np.zeros((0, 0)), all_factor_names(), np.zeros((0, 0)), []

    # 行业哑变量（快照）：对所有 series 中的 code 一次性构建
    industry_dummies = await _build_industry_factors([s["code"] for s in series])

    # 按日期对齐构建截面：收集所有日期，取公共区间内每日的因子值+收益
    date_to_rows = defaultdict(lambda: {"rows": [], "rets": [], "codes": [], "d": ""})
    all_dates = sorted({k["date"] for s in stock_data for k in s.get("kline", []) if k.get("date")})
    if len(all_dates) < 30:
        return np.zeros((0, 0)), all_factor_names(), np.zeros((0, 0)), []

    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    for s in series:
        kline = next((st["kline"] for st in stock_data if st["code"] == s["code"]), [])
        ind_row = industry_dummies.get(s["code"], [])
        for ki, kbar in enumerate(kline):
            d = kbar.get("date", "")
            if d not in date_to_idx:
                continue
            dt = date_to_idx[d]
            if dt < 25 or dt + n >= len(all_dates):
                continue
            m = series_at(s["mom"], ki)
            v = series_at(s["vol"], ki)
            if m is None or v is None or s["closes"][ki] == 0:
                continue
            row = [m, v, s["turnover"], s["ep"], s["bp"], s["size"] or 0.0] + ind_row
            date_to_rows[d]["rows"].append(row)
            date_to_rows[d]["rets"].append(s["closes"][ki + n] / s["closes"][ki] - 1.0)
            date_to_rows[d]["codes"].append(s["code"])
            date_to_rows[d]["d"] = d

    betas = []
    last_X = np.zeros((0, len(all_factor_names())))
    last_codes: list[str] = []
    for d in sorted(date_to_rows):
        entry = date_to_rows[d]
        if len(entry["rows"]) < 10:
            continue
        raw = np.array(entry["rows"], dtype=np.float64)
        X_t_style = _zscore_cols(raw[:, :len(STYLE_FACTORS)])
        if INDUSTRY_FACTORS:
            X_t_ind = raw[:, len(STYLE_FACTORS):]
            X_t = np.column_stack([X_t_style, X_t_ind])
        else:
            X_t = X_t_style
        betas.append(cross_section_regression(X_t, np.array(entry["rets"], dtype=np.float64)))
        last_X, last_codes = X_t, entry["codes"]

    if not betas or last_X.size == 0:
        return np.zeros((0, 0)), all_factor_names(), np.zeros((0, 0)), []
    return np.array(betas, dtype=np.float64), all_factor_names(), last_X, last_codes


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

    因子维度对齐：_stock_style_row 仅重建风格因子行（6 列），因此逐股残差仅用
    factor_returns 的风格因子部分（前 6 个元素），行业因子（快照常量）不参与
    时序残差计算。
    """
    if X.size == 0 or not codes:
        return np.zeros(0)
    n_style = len(STYLE_FACTORS)
    style_fr = factor_returns[:n_style] if factor_returns.shape[0] >= n_style else factor_returns
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
        if Xt.shape[1] != style_fr.shape[0]:
            continue
        resid = rt - Xt @ style_fr
        sigmas[i] = float(np.var(resid, ddof=Xt.shape[1] + 1))
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
