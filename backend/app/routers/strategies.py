"""策略 / 自定义因子 / 回测存档 持久化路由。"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import db
from . import selection as sel

router = APIRouter(prefix="/api", tags=["strategies"])

VALID_KINDS = ("select", "backtest", "regression")


# ---------------- saved_strategies ----------------

class StrategyBody(BaseModel):
    name: str
    kind: str
    config: dict


@router.get("/strategies")
def list_strategies():
    return db.list_strategies()


@router.post("/strategies")
def save_strategy(body: StrategyBody):
    if body.kind not in VALID_KINDS:
        raise HTTPException(400, f"kind 必须为 {VALID_KINDS}")
    if not body.name.strip():
        raise HTTPException(400, "name 不能为空")
    return db.create_strategy(body.name.strip(), body.kind, body.config)


@router.delete("/strategies/{sid}")
def remove_strategy(sid: int):
    if not db.delete_strategy(sid):
        raise HTTPException(404, "策略不存在")
    return {"ok": True}


@router.post("/strategies/{sid}/run")
async def run_strategy(sid: int):
    s = db.get_strategy(sid)
    if not s:
        raise HTTPException(404, "策略不存在")
    cfg = s.get("config") or {}
    kind = s["kind"]
    try:
        if kind == "backtest":
            return await sel.run_backtest(sel.BacktestBody(**cfg))
        if kind == "select":
            return await sel.run_select(sel.SelectBody(**cfg))
    except TypeError as e:
        raise HTTPException(400, f"策略配置字段不匹配: {e}")
    raise HTTPException(400, f"暂不支持一键运行 kind={kind}")


# ---------------- user_factors ----------------

class UserFactorBody(BaseModel):
    name: str
    kind: str = "composite"
    definition: dict


@router.get("/user-factors")
def list_user_factors():
    return db.list_user_factors()


@router.post("/user-factors")
def save_user_factor(body: UserFactorBody):
    if body.kind != "composite":
        raise HTTPException(400, "目前仅支持 composite 类型自定义因子")
    if not body.name.strip():
        raise HTTPException(400, "name 不能为空")
    factors = (body.definition or {}).get("factors") or []
    if not factors:
        raise HTTPException(400, "definition.factors 不能为空")
    return db.create_user_factor(body.name.strip(), body.kind, body.definition)


@router.delete("/user-factors/{fid}")
def remove_user_factor(fid: int):
    if not db.delete_user_factor(fid):
        raise HTTPException(404, "自定义因子不存在")
    return {"ok": True}


# ---------------- backtest_runs ----------------

class BacktestRunBody(BaseModel):
    strategyId: int | None = None
    config: dict
    metrics: dict
    reportPath: str | None = None


@router.get("/backtest-runs")
def list_runs(limit: int = 50):
    return db.list_backtest_runs(max(1, min(limit, 200)))


@router.post("/backtest-runs")
def save_run(body: BacktestRunBody):
    return db.create_backtest_run(body.strategyId, body.config, body.metrics, body.reportPath)


@router.delete("/backtest-runs/{rid}")
def remove_run(rid: int):
    if not db.delete_backtest_run(rid):
        raise HTTPException(404, "回测记录不存在")
    return {"ok": True}
