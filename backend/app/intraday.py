"""分钟级回测：用分钟 K 线驱动，支持盘中信号 + 止盈止损。

简化的单股分钟回测：给定一只股票 + 分钟 K 线，按信号进场、按止盈/止损/收盘平仓。
不做组合层（分钟级组合优化另接 portfolio_opt），聚焦日内执行评估。

修复：
- 加交易成本（佣金/印花税/滑点），旧版零成本
- T+1 约束：当日买入当日不能卖出（A 股规则），最早次日平仓
- maxTrades 改为真正的单日上限（旧版实为全程上限）
- 年化口径：每笔收益 per-trade Sharpe（旧版 ×960 期/年无依据）
"""
import datetime
from dataclasses import dataclass, field

from . import adapters
from .factors import mean, std, sharpe_ratio


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
    max_trades: int = 10        # 单日最大交易数（按日期重置）
    commissionRate: float = 0.00025   # 佣金（万 2.5，单边）
    stampDuty: float = 0.001          # 印花税（千 1，卖出单边）
    slippage: float = 0.001           # 滑点（单边）
    applyCost: bool = True


def _bar_date(bar: dict) -> str:
    """从分钟 K 线取日期（YYYY-MM-DD）。"""
    dt = bar.get("datetime") or bar.get("date") or ""
    return dt[:10]


async def run_intraday_backtest(cfg: IntradayConfig) -> dict:
    """单股分钟级回测（含成本 + T+1）。"""
    kline = await adapters.fetch_minute_kline(cfg.code, cfg.period, cfg.count)
    if len(kline) < cfg.signal_lookback + 2:
        return {"trades": [], "metrics": {}, "error": "分钟数据不足"}

    trades: list[IntradayTrade] = []
    position = None  # {"entry_time", "entry_price", "shares", "entry_date"}
    closes = [k["close"] for k in kline]

    cost_buy = (cfg.commissionRate + cfg.slippage) if cfg.applyCost else 0.0
    cost_sell = (cfg.commissionRate + cfg.stampDuty + cfg.slippage) if cfg.applyCost else 0.0
    trades_today: dict[str, int] = {}

    def _exit(position, bar, price, reason):
        entry_amount = position["entry_price"] * position["shares"]
        exit_amount = price * position["shares"]
        cost = cost_buy * entry_amount + cost_sell * exit_amount
        pnl = (price - position["entry_price"]) * position["shares"] - cost
        trades.append(IntradayTrade(
            position["entry_time"], position["entry_price"],
            bar.get("datetime") or bar.get("date", ""), price,
            position["shares"], pnl, reason))

    for i in range(cfg.signal_lookback, len(kline)):
        bar = kline[i]
        price = bar["close"]
        bar_date = _bar_date(bar)

        if position:
            # T+1：当日买入当日不能卖出，最早次日
            can_exit = bar_date != position["entry_date"]
            if can_exit:
                if price / position["entry_price"] - 1 >= cfg.take_profit:
                    _exit(position, bar, price, "take_profit")
                    position = None
                elif price / position["entry_price"] - 1 <= cfg.stop_loss:
                    _exit(position, bar, price, "stop_loss")
                    position = None

        if not position and trades_today.get(bar_date, 0) < cfg.max_trades:
            lookback_closes = closes[i - cfg.signal_lookback: i]
            if len(lookback_closes) >= 2 and lookback_closes[0] > 0:
                mom = price / lookback_closes[0] - 1
                if mom >= cfg.entry_threshold:
                    position = {
                        "entry_time": bar.get("datetime") or bar.get("date", ""),
                        "entry_price": price, "shares": cfg.shares_per_trade,
                        "entry_date": bar_date,
                    }
                    trades_today[bar_date] = trades_today.get(bar_date, 0) + 1

    if position:
        last = kline[-1]
        last_date = _bar_date(last)
        # T+1：仅当最后一根已跨日才收盘平仓，否则持仓过夜无法平（数据结束）
        if last_date != position["entry_date"]:
            _exit(position, last, last["close"], "close")
            position = None

    returns = [t.pnl / (t.entry_price * t.shares) for t in trades if t.entry_price * t.shares > 0]
    metrics = {
        "nTrades": len(trades),
        "winRate": sum(1 for r in returns if r > 0) / len(returns) if returns else 0.0,
        "totalPnl": sum(t.pnl for t in trades),
        "avgPnl": mean(returns) if returns else 0.0,
        # per-trade Sharpe（periods_per_year=1，不年化；旧版 ×960 无依据）
        "sharpe": sharpe_ratio(returns, periods_per_year=1) if returns else 0.0,
        "costRate": (cost_buy + cost_sell) if cfg.applyCost else 0.0,
        "applyCost": cfg.applyCost,
        "tPlus1": True,
    }
    return {
        "trades": [t.__dict__ for t in trades],
        "metrics": metrics,
        "nBars": len(kline),
        "period": cfg.period,
    }
