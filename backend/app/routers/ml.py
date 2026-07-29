"""机器学习路由：构建数据集 → 时序 CV 评估 → 训练并落盘模型。"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import ml, jobs

router = APIRouter(prefix="/api/ml", tags=["ml"])


class EvalBody(BaseModel):
    board: str = "all"
    poolSize: int = 100
    n: int = 5
    hist: int = 240
    modelType: str = "gbdt"
    nSplits: int = 5
    gap: int = 5


@router.post("/evaluate")
async def evaluate(body: EvalBody):
    """同步评估（小数据集）。大数据集建议走 /api/jobs 异步提交。"""
    try:
        dataset = await ml.build_dataset(body.board, body.poolSize, body.n, body.hist)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(502, f"数据集构建失败: {e}")
    try:
        result = ml.evaluate_dataset(dataset, body.modelType, body.nSplits, body.gap)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return result


@router.post("/train")
async def train(body: EvalBody):
    """构建数据集 + 训练最终模型并落盘。"""
    try:
        dataset = await ml.build_dataset(body.board, body.poolSize, body.n, body.hist)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(502, f"数据集构建失败: {e}")
    eval_result = ml.evaluate_dataset(dataset, body.modelType, body.nSplits, body.gap)
    meta = ml.train_final_model(dataset, body.modelType)
    return {"model": meta, "evaluation": eval_result}


@router.get("/models")
def list_models():
    return ml.list_models()


@router.delete("/models/{mid}")
def delete_model(mid: str):
    if not ml.delete_model(mid):
        raise HTTPException(404, "模型不存在")
    return {"ok": True}
