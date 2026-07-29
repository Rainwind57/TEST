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


def build_style_matrix(stock_data: list[dict]) -> tuple[np.ndarray, list[str], list[str]]:
    """构建风格因子暴露矩阵。

    stock_data: [{code, kline, quote}] 每只股票的 K 线 + 快照行情。
    返回 (X, codes, factor_names)，X 为 z-score 后的暴露矩阵。
    """
    rows = []
    codes = []
    for s in stock_data:
        kline = s.get("kline", [])
        quote = s.get("quote", {})
        if len(kline) < 25:
            continue
        arr = kline_to_arrays(kline)
        i = len(kline) - 1
        mom = series_at(compute_factor_series("momentum", arr), i)
        vol = series_at(compute_factor_series("volatility", arr), i)
        turnover = quote.get("turnover")
        pe = quote.get("pe")
        pb = quote.get("pb")
        ep = (1.0 / pe) if pe else None
        bp = (1.0 / pb) if pb else None
        if any(v is None for v in [mom, vol, turnover, ep, bp]):
            continue
        rows.append([mom, vol, turnover, ep, bp])
        codes.append(s["code"])

    if len(rows) < 3:
        return np.zeros((0, 0)), [], STYLE_FACTORS

    X = np.array(rows, dtype=np.float64)
    Xz = np.zeros_like(X)
    for c in range(X.shape[1]):
        col = X[:, c]
        valid = col[~np.isnan(col)]
        if len(valid) < 2:
            continue
        m, s = valid.mean(), valid.std()
        if s == 0:
            continue
        Xz[:, c] = (col - m) / s
    return Xz, codes, STYLE_FACTORS


def factor_covariance(X: np.ndarray) -> np.ndarray:
    """风格因子协方差矩阵。"""
    if X.size == 0:
        return np.zeros((0, 0))
    return np.cov(X, rowvar=False)


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
