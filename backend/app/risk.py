"""Barra 风格风险模型：多因子方差分解归因。

简化版（不依赖外部 Barra 数据）：
- 风格因子：动量、波动率、市值、估值（PE/PB 倒数）、换手率
- 对组合收益做因子回归，分解为 风格贡献 + 残差（特质风险）
- 输出各风格因子暴露、贡献、残差波动
"""
import numpy as np
from .factors import mean, std, zscore
from .numpy_factors import kline_to_arrays, compute_factor_series, series_at


STYLE_FACTORS = ["momentum", "volatility", "turnover", "ep", "bp"]


def build_style_panel(stock_data: list[dict], n: int = 1) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Fama-MacBeth 时序面板：对每个历史截面回归得因子收益 beta_t。

    返回 (betas, factor_names, last_X, last_codes)：
    - betas: (n_periods, n_factors) 因子收益时序，用于估计因子收益协方差
    - last_X / last_codes: 最新截面暴露矩阵，用于组合归因
    turnover/ep/bp 无历史时序，用最新快照近似（已知简化）。
    """
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
        if any(v is None for v in [mom, vol, turnover, pe, pb]) or pe == 0 or pb == 0:
            continue
        series.append({"code": s["code"], "closes": arr["close"], "mom": mom, "vol": vol,
                       "turnover": turnover, "ep": 1.0 / pe, "bp": 1.0 / pb})
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
            rows.append([m, v, s["turnover"], s["ep"], s["bp"]])
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


def factor_covariance(factor_returns_panel: np.ndarray) -> np.ndarray:
    """因子收益时序协方差矩阵（Fama-MacBeth beta 时序估计）。

    输入应为 (n_periods, n_factors) 的因子收益时序。旧版误用截面暴露矩阵 np.cov(X)
    冒充因子收益协方差——对象完全错误（截面暴露共变 ≠ 因子收益时序共变）。
    """
    if factor_returns_panel.size == 0:
        return np.zeros((0, 0))
    if factor_returns_panel.ndim < 2 or factor_returns_panel.shape[0] < 2:
        k = factor_returns_panel.shape[-1] if factor_returns_panel.ndim >= 1 else 0
        return np.zeros((k, k))
    return np.cov(factor_returns_panel, rowvar=False)


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
