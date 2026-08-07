"""参数寻优路由：Optuna 贝叶斯搜索回测参数，Walk-Forward IS/OOS 评估。"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

import logging

from .. import optimize, jobs

logger = logging.getLogger(__name__)
from .auth import require_user_id

router = APIRouter(prefix="/api/optimize", tags=["optimize"])


class OptimizeBody(BaseModel):
    board: str = "all"
    boards: list[str] | None = None   # 多板块 OR 组合，优先于 board
    poolSize: int = 60
    factor: str = "momentum"
    modelId: str | None = None      # 指定时对 ML 模型策略寻优（与技术因子二选一）
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
    # 寻优结果落盘：最优参数可被"保存策略→回测→组合→风险"链路复用
    from .. import artifacts
    try:
        meta = artifacts.save_artifact("optimize", {
            "baseConfig": body.model_dump(),
            "bestParams": result.get("bestParams", {}),
            "isMetrics": result.get("isMetrics"),
            "oosMetrics": result.get("oosMetrics"),
            "splitDate": result.get("splitDate"),
        }, name=f"寻优-{body.factor or body.modelId}")
        result["artifact"] = meta
    except Exception as e:
        logger.warning("寻优结果落盘失败: %s", e)
    return result


@router.post("/save-strategy")
def save_as_strategy(body: SaveStrategyBody, uid: int = Depends(require_user_id)):
    """把最优参数回写为策略。"""
    try:
        return optimize.save_best_as_strategy(body.baseConfig, body.bestParams, body.name, user_id=uid)
    except Exception as e:
        raise HTTPException(400, f"保存失败: {e}")
