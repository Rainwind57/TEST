"""策略 / 自定义因子 / 回测存档 持久化路由。"""
from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel

from .. import db, intraday
from ..auth import get_user_id_from_auth
from .auth import require_user_id
from . import selection as sel
from .intraday import IntradayBody
from . import ml as ml_router

router = APIRouter(prefix="/api", tags=["strategies"])

VALID_KINDS = ("select", "backtest", "regression", "intraday", "ml")


def _uid(request: Request) -> int:
    return get_user_id_from_auth(request.headers.get("Authorization")) or 0


# ---------------- saved_strategies ----------------

class StrategyBody(BaseModel):
    name: str
    kind: str
    config: dict


@router.get("/strategies")
def list_strategies(request: Request):
    return db.list_strategies(_uid(request))


@router.post("/strategies")
def save_strategy(body: StrategyBody, uid: int = Depends(require_user_id)):
    if body.kind not in VALID_KINDS:
        raise HTTPException(400, f"kind 必须为 {VALID_KINDS}")
    if not body.name.strip():
        raise HTTPException(400, "name 不能为空")
    return db.create_strategy(body.name.strip(), body.kind, body.config, uid)


@router.delete("/strategies/{sid}")
def remove_strategy(sid: int, uid: int = Depends(require_user_id)):
    if not db.delete_strategy(sid, uid):
        raise HTTPException(404, "策略不存在或无权操作")
    return {"ok": True}


@router.post("/strategies/{sid}/run")
async def run_strategy(sid: int, request: Request):
    s = db.get_strategy(sid)
    if not s:
        raise HTTPException(404, "策略不存在")
    if s.get("user_id") != _uid(request):
        raise HTTPException(403, "无权运行该策略")
    cfg = s.get("config") or {}
    kind = s["kind"]
    try:
        if kind == "backtest":
            return await sel.run_backtest(sel.BacktestBody(**cfg))
        if kind == "select":
            return await sel.run_select(sel.SelectBody(**cfg))
        if kind == "regression":
            return await sel.run_factor_regression(sel.FactorRegressionBody(**cfg))
        if kind == "intraday":
            ib = IntradayBody(**cfg)
            icfg = intraday.IntradayConfig(
                code=ib.code, period=ib.period, count=ib.count,
                signal_lookback=ib.signalLookback, entry_threshold=ib.entryThreshold,
                take_profit=ib.takeProfit, stop_loss=ib.stopLoss,
                shares_per_trade=ib.sharesPerTrade, max_trades=ib.maxTrades,
                commissionRate=ib.commissionRate, stampDuty=ib.stampDuty,
                slippage=ib.slippage, applyCost=ib.applyCost,
            )
            return await intraday.run_intraday_backtest(icfg)
        if kind == "ml":
            from .ml import MLBacktestBody
            return await ml_router.ml_backtest(MLBacktestBody(**cfg))
    except (TypeError, ValueError) as e:
        raise HTTPException(400, f"策略配置字段不匹配: {e}")
    raise HTTPException(400, f"暂不支持一键运行 kind={kind}")


# ---------------- user_factors ----------------

class UserFactorBody(BaseModel):
    name: str
    kind: str = "composite"
    definition: dict


@router.get("/user-factors")
def list_user_factors(request: Request):
    return db.list_user_factors(_uid(request))


@router.post("/user-factors")
def save_user_factor(body: UserFactorBody, uid: int = Depends(require_user_id)):
    if body.kind not in ("composite", "expression", "model"):
        raise HTTPException(400, "目前仅支持 composite / expression / model 类型自定义因子")
    if not body.name.strip():
        raise HTTPException(400, "name 不能为空")
    definition = body.definition or {}
    if body.kind == "composite" and not definition.get("factors"):
        raise HTTPException(400, "composite 类型需要 definition.factors")
    if body.kind == "expression":
        from .. import factor_expr
        ok, err = factor_expr.validate_expression(definition.get("expression") or "")
        if not ok:
            raise HTTPException(400, f"表达式非法: {err}")
    if body.kind == "model" and not definition.get("modelId"):
        raise HTTPException(400, "model 类型需要 definition.modelId")
    return db.create_user_factor(body.name.strip(), body.kind, body.definition, uid)


@router.delete("/user-factors/{fid}")
def remove_user_factor(fid: int, uid: int = Depends(require_user_id)):
    if not db.delete_user_factor(fid, uid):
        raise HTTPException(404, "自定义因子不存在或无权操作")
    return {"ok": True}


# ---------------- backtest_runs ----------------

class BacktestRunBody(BaseModel):
    strategyId: int | None = None
    config: dict
    metrics: dict
    reportPath: str | None = None


@router.get("/backtest-runs")
def list_runs(request: Request, limit: int = 50):
    return db.list_backtest_runs(max(1, min(limit, 200)), _uid(request))


@router.post("/backtest-runs")
def save_run(body: BacktestRunBody, request: Request):
    return db.create_backtest_run(body.strategyId, body.config, body.metrics, body.reportPath, _uid(request))


@router.delete("/backtest-runs/{rid}")
def remove_run(rid: int, request: Request):
    if not db.delete_backtest_run(rid, _uid(request)):
        raise HTTPException(404, "回测记录不存在或无权操作")
    return {"ok": True}
