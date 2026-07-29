"""异步任务提交与轮询路由。

长回测/选股改为异步：POST /api/jobs 提交返回 job_id，
GET /api/jobs/{id} 轮询进度与结果。同步接口（/api/select, /api/select/backtest）保持不变。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import jobs
from . import selection as sel

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class JobSubmitBody(BaseModel):
    kind: str  # "select" | "backtest" | "factor-regression"
    config: dict


@router.post("")
async def submit_job(body: JobSubmitBody):
    if body.kind not in ("select", "backtest", "factor-regression"):
        raise HTTPException(400, "kind 必须为 select / backtest / factor-regression")

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

    try:
        cfg = sel.FactorRegressionBody(**body.config)
    except Exception as e:
        raise HTTPException(400, f"config 字段不匹配: {e}")
    jid = jobs.create_job("factor-regression", body.config)
    jobs.submit(jid, sel.run_factor_regression(cfg))
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
