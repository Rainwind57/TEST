"""参数寻优路由：Optuna 贝叶斯搜索回测参数，Walk-Forward IS/OOS 评估。"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import optimize, jobs

router = APIRouter(prefix="/api/optimize", tags=["optimize"])


class OptimizeBody(BaseModel):
    board: str = "all"
    poolSize: int = 60
    factor: str = "momentum"
    groups: int = 5
    n: int = 5
    hist: int = 180
    commissionRate: float = 0.00025
    stampDuty: float = 0.001
    slippage: float = 0.001
    benchmark: str = "none"
    applyCost: bool = True
    nTrials: int = 30


class SaveStrategyBody(BaseModel):
    name: str
    baseConfig: dict
    bestParams: dict


@router.post("/backtest")
def optimize_backtest(body: OptimizeBody):
    """同步寻优（小试验数）。大数据建议走 /api/jobs。"""
    try:
        result = optimize.optimize_backtest(body.model_dump(), body.nTrials)
    except Exception as e:
        raise HTTPException(502, f"寻优失败: {e}")
    return result


@router.post("/save-strategy")
def save_as_strategy(body: SaveStrategyBody):
    """把最优参数回写为策略。"""
    try:
        return optimize.save_best_as_strategy(body.baseConfig, body.bestParams, body.name)
    except Exception as e:
        raise HTTPException(400, f"保存失败: {e}")
