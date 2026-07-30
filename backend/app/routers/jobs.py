"""异步任务提交与轮询路由。

长回测/选股/ML 训练/参数寻优改为异步：POST /api/jobs 提交返回 job_id，
GET /api/jobs/{id} 轮询进度与结果。同步接口保持不变。
"""
import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import jobs, ml, optimize
from . import selection as sel
from .ml import EvalBody
from .optimize import OptimizeBody

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

_VALID_KINDS = ("select", "backtest", "factor-regression",
                "ml-evaluate", "ml-train", "ml-backtest", "optimize")


class JobSubmitBody(BaseModel):
    kind: str  # 见 _VALID_KINDS
    config: dict


@router.post("")
async def submit_job(body: JobSubmitBody):
    if body.kind not in _VALID_KINDS:
        raise HTTPException(400, f"kind 必须为 {list(_VALID_KINDS)}")

    if body.kind == "backtest":
        try:
            cfg = sel.BacktestBody(**body.config)
        except Exception as e:
            raise HTTPException(400, f"config 字段不匹配: {e}")
        jid = jobs.create_job("backtest", body.config)
        jobs.submit(jid, sel.run_backtest(cfg))
        return {"jobId": jid, "status": "pending"}

    if body.kind == "select":
        try:
            cfg = sel.SelectBody(**body.config)
        except Exception as e:
            raise HTTPException(400, f"config 字段不匹配: {e}")
        jid = jobs.create_job("select", body.config)
        jobs.submit(jid, sel.run_select(cfg))
        return {"jobId": jid, "status": "pending"}

    if body.kind == "factor-regression":
        try:
            cfg = sel.FactorRegressionBody(**body.config)
        except Exception as e:
            raise HTTPException(400, f"config 字段不匹配: {e}")
        jid = jobs.create_job("factor-regression", body.config)
        jobs.submit(jid, sel.run_factor_regression(cfg))
        return {"jobId": jid, "status": "pending"}

    if body.kind in ("ml-evaluate", "ml-train", "ml-backtest"):
        return _submit_ml_job(body.kind, body.config)

    # optimize：同步函数（内部自建 event loop），放线程池避免阻塞主 loop
    try:
        cfg = OptimizeBody(**body.config)
    except Exception as e:
        raise HTTPException(400, f"config 字段不匹配: {e}")
    config = body.config
    n_trials = cfg.nTrials
    jid = jobs.create_job("optimize", config)

    async def _run_optimize():
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, optimize.optimize_backtest, config, n_trials)

    jobs.submit(jid, _run_optimize())
    return {"jobId": jid, "status": "pending"}


def _submit_ml_job(kind: str, config: dict) -> dict:
    """ML 评估/训练/回测异步提交。"""
    try:
        cfg = EvalBody(**config) if kind != "ml-backtest" else None
        ml_backtest_cfg = config if kind == "ml-backtest" else None
    except Exception as e:
        raise HTTPException(400, f"config 字段不匹配: {e}")
    jid = jobs.create_job(kind, config)

    if kind == "ml-evaluate":
        async def _run():
            dataset = await ml.build_dataset(cfg.board, cfg.poolSize, cfg.n, cfg.hist, use_snapshot=cfg.useSnapshot)
            return ml.evaluate_dataset(dataset, cfg.modelType, cfg.nSplits, cfg.gap)
        jobs.submit(jid, _run())
    elif kind == "ml-train":
        async def _run():
            dataset = await ml.build_dataset(cfg.board, cfg.poolSize, cfg.n, cfg.hist, use_snapshot=cfg.useSnapshot)
            ev = ml.evaluate_dataset(dataset, cfg.modelType, cfg.nSplits, cfg.gap)
            meta = ml.train_final_model(dataset, cfg.modelType)
            return {"model": meta, "evaluation": ev}
        jobs.submit(jid, _run())
    else:  # ml-backtest
        from .ml import MLBacktestBody
        try:
            bt_cfg = MLBacktestBody(**config)
        except Exception as e:
            raise HTTPException(400, f"config 字段不匹配: {e}")
        jobs.submit(jid, ml.backtest_model(
            bt_cfg.modelId, bt_cfg.board, bt_cfg.poolSize, bt_cfg.groups, bt_cfg.n,
            bt_cfg.hist, bt_cfg.commissionRate, bt_cfg.stampDuty, bt_cfg.slippage,
            bt_cfg.benchmark, bt_cfg.applyCost))
    return {"jobId": jid, "status": "pending"}


@router.get("")
def list_jobs(limit: int = 50):
    return jobs.list_jobs(max(1, min(limit, 200)))


@router.get("/{jid}")
def get_job(jid: str):
    j = jobs.get_job(jid)
    if not j:
        raise HTTPException(404, "任务不存在")
    return j


@router.delete("/{jid}")
def cancel_job(jid: str):
    if not jobs.cancel(jid):
        raise HTTPException(404, "任务不存在或已完成")
    return {"ok": True}
