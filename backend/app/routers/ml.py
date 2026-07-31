"""机器学习路由：构建数据集 → 时序 CV 评估 → 训练并落盘模型。"""
import asyncio
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from .. import ml, jobs
from .auth import require_user_id

router = APIRouter(prefix="/api/ml", tags=["ml"])


class EvalBody(BaseModel):
    board: str = "all"
    poolSize: int = 100
    n: int = 5
    hist: int = 240
    modelType: str = "gbdt"
    nSplits: int = 5
    gap: int = 5
    useSnapshot: bool = False  # 追加 pe/pb/turniture 快照特征（含前视风险，探索用）
    nTrials: int = 30  # ml-optimize 用（Optuna 试验数）


@router.post("/evaluate")
async def evaluate(body: EvalBody):
    """同步评估（小数据集）。大数据集建议走 /api/jobs 异步提交。"""
    try:
        dataset = await ml.build_dataset(body.board, body.poolSize, body.n, body.hist, use_snapshot=body.useSnapshot)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(502, f"数据集构建失败: {e}")
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, ml.evaluate_dataset, dataset, body.modelType, body.nSplits, body.gap)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return result

@router.post("/train")
async def train(body: EvalBody):
    """构建数据集 + 训练最终模型并落盘。"""
    try:
        dataset = await ml.build_dataset(body.board, body.poolSize, body.n, body.hist, use_snapshot=body.useSnapshot)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(502, f"数据集构建失败: {e}")
    # CPU 密集同步函数放线程池，避免阻塞事件循环（旧版训练期间全站请求卡死）
    loop = asyncio.get_running_loop()
    try:
        eval_result = await loop.run_in_executor(None, ml.evaluate_dataset, dataset, body.modelType, body.nSplits, body.gap)
    except ValueError as e:
        raise HTTPException(422, str(e))
    meta = await loop.run_in_executor(None, ml.train_final_model, dataset, body.modelType)
    return {"model": meta, "evaluation": eval_result}


@router.get("/models")
def list_models():
    return ml.list_models()


class OptimizeMlBody(BaseModel):
    board: str = "all"
    poolSize: int = 100
    n: int = 5
    hist: int = 240
    modelType: str = "lightgbm"  # 默认启用已装的 lightgbm（旧版仅 gbdt）
    nSplits: int = 5
    gap: int = 5
    nTrials: int = 30
    useSnapshot: bool = False


@router.post("/optimize")
async def optimize(body: OptimizeMlBody, uid: int = Depends(require_user_id)):
    """ML 超参寻优（Optuna + Walk-Forward OOS Sharpe）。

    旧版 Optuna 仅接因子回测（optimize.optimize_backtest），ML _build_model
    硬编码超参、无法自动寻优；此端点打通 ML 调参闭环，并落盘实验记录。
    大数据集建议走 /api/jobs（kind=ml-optimize）。
    """
    try:
        dataset = await ml.build_dataset(body.board, body.poolSize, body.n, body.hist, use_snapshot=body.useSnapshot)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(502, f"数据集构建失败: {e}")
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None, ml.optimize_model, dataset, body.modelType, body.nSplits, body.gap, body.nTrials)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return result


@router.delete("/models/{mid}")
def delete_model(mid: str, uid: int = Depends(require_user_id)):
    if not ml.delete_model(mid):
        raise HTTPException(404, "模型不存在")
    return {"ok": True}


class ScoreBody(BaseModel):
    modelId: str
    board: str = "all"
    poolSize: int = 100


@router.post("/score")
async def score(body: ScoreBody):
    """用落盘模型对候选池最新截面打分（打通 ML→选股：结果可加入自选/买入模拟盘）。"""
    try:
        return await ml.score_latest(body.modelId, body.board, body.poolSize)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(502, f"打分失败: {e}")


class MLBacktestBody(BaseModel):
    modelId: str
    board: str = "all"
    poolSize: int = 60
    groups: int = 5
    n: int = 5
    hist: int = 180
    commissionRate: float = 0.00025
    stampDuty: float = 0.001
    slippage: float = 0.001
    benchmark: str = "none"
    applyCost: bool = True


@router.post("/backtest")
async def ml_backtest(body: MLBacktestBody):
    """ML 信号分层回测（打通 ML→回测，响应结构与 /api/select/backtest 一致，前端图表零成本复用）。"""
    try:
        return await ml.backtest_model(
            body.modelId, body.board, body.poolSize, body.groups, body.n, body.hist,
            body.commissionRate, body.stampDuty, body.slippage, body.benchmark, body.applyCost,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(502, f"ML 回测失败: {e}")
