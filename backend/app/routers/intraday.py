"""分钟级回测路由：单股日内策略回测。"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import intraday

router = APIRouter(prefix="/api/intraday", tags=["intraday"])


class IntradayBody(BaseModel):
    code: str
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


@router.post("/backtest")
async def backtest(body: IntradayBody):
    cfg = intraday.IntradayConfig(
        code=body.code, period=body.period, count=body.count,
        signal_lookback=body.signalLookback, entry_threshold=body.entryThreshold,
        take_profit=body.takeProfit, stop_loss=body.stopLoss,
        shares_per_trade=body.sharesPerTrade, max_trades=body.maxTrades,
        commissionRate=body.commissionRate, stampDuty=body.stampDuty,
        slippage=body.slippage, applyCost=body.applyCost,
    )
    try:
        return await intraday.run_intraday_backtest(cfg)
    except Exception as e:
        raise HTTPException(502, f"分钟级回测失败: {e}")
