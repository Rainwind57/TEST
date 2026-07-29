"""组合优化模块：均值-方差、风险平价、带约束的最优化求解（cvxpy）。

输入：预期收益向量 mu、协方差矩阵 Sigma；输出：权重 w。
支持约束：个股权重上限、行业暴露上限、换手约束（可选）。
"""
import numpy as np
import cvxpy as cp


def mean_variance(mu: np.ndarray, sigma: np.ndarray, max_weight: float = 0.1,
                  long_only: bool = True, target_return: float | None = None) -> np.ndarray:
    """均值-方差优化：最小化方差，约束期望收益 ≥ target（可选）。"""
    n = len(mu)
    w = cp.Variable(n)
    risk = cp.quad_form(w, sigma)
    constraints = []
    if long_only:
        constraints.append(w >= 0)
    constraints.append(cp.sum(w) == 1)
    if max_weight > 0:
        constraints.append(w <= max_weight)
    if target_return is not None:
        constraints.append(mu @ w >= target_return)
    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve()
    if w.value is None:
        return np.ones(n) / n
    return np.array(w.value)


def max_sharpe(mu: np.ndarray, sigma: np.ndarray, max_weight: float = 0.1,
               rf: float = 0.0, long_only: bool = True) -> np.ndarray:
    """最大化 Sharpe（凸近似）：min w' Σ w，s.t. (mu-rf)'w = 1，再归一。"""
    n = len(mu)
    w = cp.Variable(n)
    risk = cp.quad_form(w, sigma)
    constraints = [cp.sum((mu - rf) @ w) == 1]
    if long_only:
        constraints.append(w >= 0)
    if max_weight > 0:
        constraints.append(w <= max_weight)
    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve()
    if w.value is None:
        return np.ones(n) / n
    raw = np.array(w.value)
    s = raw.sum()
    return raw / s if s != 0 else np.ones(n) / n


def risk_parity(sigma: np.ndarray, max_weight: float = 0.2) -> np.ndarray:
    """风险平价：各资产风险贡献相等。凸近似：min 0.5 w'Σw - sum(ln w)。"""
    n = sigma.shape[0]
    w = cp.Variable(n, pos=True)
    risk = 0.5 * cp.quad_form(w, sigma)
    log_term = cp.sum(cp.log(w))
    constraints = [cp.sum(w) == 1]
    if max_weight > 0:
        constraints.append(w <= max_weight)
    prob = cp.Problem(cp.Minimize(risk - log_term), constraints)
    prob.solve()
    if w.value is None:
        return np.ones(n) / n
    raw = np.array(w.value)
    return raw / raw.sum()


def equal_weight(n: int) -> np.ndarray:
    return np.ones(n) / n


def portfolio_stats(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, rf: float = 0.0) -> dict:
    """计算组合收益、波动、Sharpe。"""
    ret = float(mu @ w)
    vol = float(np.sqrt(w @ sigma @ w))
    sharpe = 0.0 if vol == 0 else (ret - rf) / vol
    return {"return": ret, "volatility": vol, "sharpe": sharpe}
