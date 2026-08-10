"""分钟级回测路由：单股日内策略回测。"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from .. import intraday
from .auth import require_user_id

router = APIRouter(prefix="/api/intraday", tags=["intraday"])


class IntradayBody(BaseModel):
    code: str = ""
    codes: list[str] | None = None   # 组合级分钟回测：传 codes 时对多标的批量执行
    period: str = "5"
    count: int = 240
    signalLookback: int = 10
    entryThreshold: float = 0.005
    takeProfit: float = 0.02
    stopLoss: float = -0.01
    sharesPerTrade: int = 100
    maxTrades: int = 10
    commissionRate: float = 0.00025
    stampDuty: float = 0.001
    slippage: float = 0.001
    applyCost: bool = True
    saveArtifact: bool = False   # 回测结果落盘为中间结果


@router.post("/backtest")
async def backtest(body: IntradayBody, uid: int = Depends(require_user_id)):
    if not body.code and not body.codes:
        raise HTTPException(400, "code（单股）或 codes（组合）至少提供一个")
    if body.takeProfit <= 0:
        raise HTTPException(400, "takeProfit 必须 > 0（例如 0.02 表示 +2% 止盈）")
    if body.stopLoss >= 0:
        raise HTTPException(400, "stopLoss 必须 < 0（例如 -0.01 表示 -1% 止损）")
    cfg = intraday.IntradayConfig(
        code=body.code, period=body.period, count=body.count,
        signal_lookback=body.signalLookback, entry_threshold=body.entryThreshold,
        take_profit=body.takeProfit, stop_loss=body.stopLoss,
        shares_per_trade=body.sharesPerTrade, max_trades=body.maxTrades,
        commissionRate=body.commissionRate, stampDuty=body.stampDuty,
        slippage=body.slippage, applyCost=body.applyCost,
    )
    try:
        if body.codes:
            result = await intraday.run_intraday_pool_backtest(body.codes, cfg)
        else:
            result = await intraday.run_intraday_backtest(cfg)
    except Exception as e:
        raise HTTPException(502, f"分钟级回测失败: {e}")
    if body.saveArtifact:
        from .. import artifacts
        meta = artifacts.save_artifact("intraday", result,
                                       name=f"分钟回测-{body.code or '组合'}")
        result["artifact"] = meta
    return result
