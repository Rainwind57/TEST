"""组合优化模块：均值-方差、风险平价、带约束的最优化求解（cvxpy）。

输入：预期收益向量 mu、协方差矩阵 Sigma；输出：权重 w。
支持约束：个股权重上限。所有求解器异常统一捕获并回退等权，避免直接 500。
"""
import numpy as np
import cvxpy as cp


def _fallback(n: int) -> np.ndarray:
    return np.ones(n) / n


def _normalize(w_raw: np.ndarray, max_weight: float = 0.0, long_only: bool = True) -> np.ndarray:
    """归一化权重；long_only=False 时按 sum(|w|) 归一化、不 clip 负权。"""
    if not long_only:
        abs_sum = float(np.abs(w_raw).sum())
        w = w_raw / abs_sum if abs_sum > 0 else _fallback(len(w_raw))
        if max_weight > 0:
            w = np.clip(w, -max_weight, max_weight)
        return w
    s = w_raw.sum()
    w = w_raw / s if s != 0 else _fallback(len(w_raw))
    w = np.clip(w, 0, None)
    s = w.sum()
    w = w / s if s != 0 else _fallback(len(w))
    if max_weight <= 0:
        return w
    for _ in range(10):
        over = w > max_weight + 1e-9
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = ~over
        if under.any() and w[under].sum() > 0:
            w[under] += excess * w[under] / w[under].sum()
        else:
            break
        s = w.sum()
        if s > 0:
            w = w / s
    return w


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
    try:
        prob.solve()
    except Exception:
        raise ValueError("组合优化求解器异常（可能矩阵奇异或约束冲突）")
    if w.value is None:
        if prob.status in ("infeasible", "infeasible_inaccurate"):
            raise ValueError(f"约束不可行：请检查 maxWeight({max_weight}) 是否过严（n×maxWeight < 1 则必无解）")
        raise ValueError(f"组合优化无解（status={prob.status}），请放宽约束或更换输入")
    return _normalize(np.array(w.value), max_weight if max_weight > 0 else 0, long_only)


def max_sharpe(mu: np.ndarray, sigma: np.ndarray, max_weight: float = 0.1,
               rf: float = 0.0, long_only: bool = True) -> np.ndarray:
    """最大化 Sharpe（凸近似）：min w'Σw s.t. (mu-rf)'w = 1，再归一化 + 投影 max_weight。"""
    n = len(mu)
    w = cp.Variable(n)
    risk = cp.quad_form(w, sigma)
    constraints = [cp.sum((mu - rf) @ w) == 1]
    if long_only:
        constraints.append(w >= 0)
    if max_weight > 0:
        constraints.append(w <= max_weight)
    prob = cp.Problem(cp.Minimize(risk), constraints)
    try:
        prob.solve()
    except Exception:
        raise ValueError("组合优化求解器异常（可能矩阵奇异或约束冲突）")
    if w.value is None:
        if prob.status in ("infeasible", "infeasible_inaccurate"):
            raise ValueError(f"约束不可行：请检查 maxWeight({max_weight}) 是否过严")
        raise ValueError(f"组合优化无解（status={prob.status}），请放宽约束或更换输入")
    return _normalize(np.array(w.value), max_weight, long_only)


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
    try:
        prob.solve()
    except Exception:
        raise ValueError("风险平价求解器异常（可能矩阵奇异或约束冲突）")
    if w.value is None:
        if prob.status in ("infeasible", "infeasible_inaccurate"):
            raise ValueError(f"约束不可行：请检查 maxWeight({max_weight}) 是否过严")
        raise ValueError(f"风险平价无解（status={prob.status}），请放宽约束或更换输入")
    return w.value / w.value.sum()


def equal_weight(n: int) -> np.ndarray:
    return np.ones(n) / n


def portfolio_stats(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, rf: float = 0.0) -> dict:
    """计算组合收益、波动、Sharpe。"""
    ret = float(mu @ w)
    vol = float(np.sqrt(w @ sigma @ w))
    sharpe = 0.0 if vol == 0 else (ret - rf) / vol
    return {"return": ret, "volatility": vol, "sharpe": sharpe}
