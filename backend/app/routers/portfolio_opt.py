"""组合优化路由：均值-方差/最大 Sharpe/风险平价，带个股权重上限约束。"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np

from .. import portfolio_opt as po

router = APIRouter(prefix="/api/portfolio-opt", tags=["portfolio-opt"])


class OptBody(BaseModel):
    mu: list[float]               # 预期收益向量
    cov: list[list[float]]        # 协方差矩阵
    method: str = "mean_variance" # mean_variance | max_sharpe | risk_parity | equal
    maxWeight: float = 0.1
    longOnly: bool = True
    rf: float = 0.0
    targetReturn: float | None = None


@router.post("")
def optimize(body: OptBody):
    mu = np.array(body.mu, dtype=np.float64)
    cov = np.array(body.cov, dtype=np.float64)
    if mu.ndim != 1 or cov.ndim != 2 or mu.shape[0] != cov.shape[0] != cov.shape[1]:
        raise HTTPException(400, "mu 与 cov 维度不匹配")

    method = body.method
    if method == "mean_variance":
        w = po.mean_variance(mu, cov, body.maxWeight, body.longOnly, body.targetReturn)
    elif method == "max_sharpe":
        w = po.max_sharpe(mu, cov, body.maxWeight, body.rf, body.longOnly)
    elif method == "risk_parity":
        w = po.risk_parity(cov, body.maxWeight)
    elif method == "equal":
        w = po.equal_weight(len(mu))
    else:
        raise HTTPException(400, f"未知方法: {method}")

    stats = po.portfolio_stats(w, mu, cov, body.rf)
    return {"weights": w.tolist(), "stats": stats, "method": method}
