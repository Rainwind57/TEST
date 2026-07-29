"""分钟级回测：用分钟 K 线驱动，支持盘中信号 + 止盈止损。

简化的单股分钟回测：给定一只股票 + 分钟 K 线，按信号进场、按止盈/止损/收盘平仓。
不做组合层（分钟级组合优化另接 portfolio_opt），聚焦日内执行评估。
"""
import asyncio
import datetime
from dataclasses import dataclass, field

from . import adapters
from .factors import mean, std, annualized_return, annualized_volatility, sharpe_ratio


@dataclass
class IntradayTrade:
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    shares: int
    pnl: float
    reason: str  # "take_profit" | "stop_loss" | "close"


@dataclass
class IntradayConfig:
    code: str
    period: str = "5"           # 1/5/15/30/60
    count: int = 240            # K 线数量
    signal_lookback: int = 10   # 进场信号回看窗口（动量）
    entry_threshold: float = 0.005  # 动量超过该值进场
    take_profit: float = 0.02   # 止盈
    stop_loss: float = -0.01    # 止损
    shares_per_trade: int = 100
    max_trades: int = 10        # 单日最大交易数


async def run_intraday_backtest(cfg: IntradayConfig) -> dict:
    """单股分钟级回测。"""
    kline = await adapters.fetch_minute_kline(cfg.code, cfg.period, cfg.count)
    if len(kline) < cfg.signal_lookback + 2:
        return {"trades": [], "metrics": {}, "error": "分钟数据不足"}

    trades: list[IntradayTrade] = []
    position = None  # {"entry_time", "entry_price", "shares"}
    closes = [k["close"] for k in kline]

    for i in range(cfg.signal_lookback, len(kline)):
        bar = kline[i]
        price = bar["close"]

        if position:
            if price / position["entry_price"] - 1 >= cfg.take_profit:
                pnl = (price - position["entry_price"]) * position["shares"]
                trades.append(IntradayTrade(
                    position["entry_time"], position["entry_price"],
                    bar["datetime"], price, position["shares"], pnl, "take_profit"))
                position = None
            elif price / position["entry_price"] - 1 <= cfg.stop_loss:
                pnl = (price - position["entry_price"]) * position["shares"]
                trades.append(IntradayTrade(
                    position["entry_time"], position["entry_price"],
                    bar["datetime"], price, position["shares"], pnl, "stop_loss"))
                position = None

        if not position and len(trades) < cfg.max_trades:
            lookback_closes = closes[i - cfg.signal_lookback: i]
            if len(lookback_closes) >= 2 and lookback_closes[0] > 0:
                mom = price / lookback_closes[0] - 1
                if mom >= cfg.entry_threshold:
                    position = {"entry_time": bar["datetime"], "entry_price": price,
                                "shares": cfg.shares_per_trade}

    if position:
        last = kline[-1]
        pnl = (last["close"] - position["entry_price"]) * position["shares"]
        trades.append(IntradayTrade(
            position["entry_time"], position["entry_price"],
            last["datetime"], last["close"], position["shares"], pnl, "close"))

    returns = [t.pnl / (t.entry_price * t.shares) for t in trades if t.entry_price * t.shares > 0]
    metrics = {
        "nTrades": len(trades),
        "winRate": sum(1 for r in returns if r > 0) / len(returns) if returns else 0.0,
        "totalPnl": sum(t.pnl for t in trades),
        "avgPnl": mean(returns) if returns else 0.0,
        "annualizedReturn": annualized_return(returns, periods_per_year=240 * 4) if returns else 0.0,
        "annualizedVolatility": annualized_volatility(returns, periods_per_year=240 * 4) if returns else 0.0,
        "sharpe": sharpe_ratio(returns, periods_per_year=240 * 4) if returns else 0.0,
    }
    return {
        "trades": [t.__dict__ for t in trades],
        "metrics": metrics,
        "nBars": len(kline),
        "period": cfg.period,
    }
