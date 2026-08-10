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
        bar_date = _bar_date(bar)

        if position:
            can_exit = bar_date != position["entry_date"]
            if can_exit:
                bar_open = bar.get("open", bar["close"])
                bar_high = bar.get("high", bar["close"])
                bar_low = bar.get("low", bar["close"])
                tp_hit = bar_high / position["entry_price"] - 1 >= cfg.take_profit
                sl_hit = bar_low / position["entry_price"] - 1 <= cfg.stop_loss
                if tp_hit and sl_hit:
                    # 同一bar同时触及止盈止损：用开盘价成交（实际可达价），不偏向任何一方
                    _exit(position, bar, bar_open, "open")
                    position = None
                elif tp_hit:
                    # 跳空穿越止盈：按实际开盘价成交，不按理想止盈价
                    exit_price = bar_open if bar_open / position["entry_price"] - 1 >= cfg.take_profit else position["entry_price"] * (1 + cfg.take_profit)
                    _exit(position, bar, exit_price, "take_profit")
                    position = None
                elif sl_hit:
                    # 跳空穿越止损：按实际开盘价成交，不按理想止损价（收益不虚高）
                    exit_price = bar_open if bar_open / position["entry_price"] - 1 <= cfg.stop_loss else position["entry_price"] * (1 + cfg.stop_loss)
                    _exit(position, bar, exit_price, "stop_loss")
                    position = None

        if not position and trades_today.get(bar_date, 0) < cfg.max_trades:
            lookback_closes = closes[i - cfg.signal_lookback: i]
            if len(lookback_closes) >= 2 and lookback_closes[0] > 0:
                mom = bar["close"] / lookback_closes[0] - 1
                if mom >= cfg.entry_threshold:
                    # 信号在 bar i 产生，成交用 bar i+1 的 open；若已是最后一根则用 close
                    next_bar = kline[i + 1] if i + 1 < len(kline) else bar
                    entry_price = next_bar.get("open", next_bar["close"])
                    position = {
                        "entry_time": next_bar.get("datetime") or next_bar.get("date", ""),
                        "entry_price": entry_price, "shares": cfg.shares_per_trade,
                        "entry_date": _bar_date(next_bar),
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
    # 跨日跳空检测（>3% 可能为除权除息，分钟线未复权）
    div_warning = None
    prev_date = None
    prev_close = None
    for bar in kline:
        d = _bar_date(bar)
        if prev_date and d != prev_date and prev_close and prev_close > 0:
            gap = abs(bar["close"] / prev_close - 1)
            if gap > 0.03:
                div_warning = f"检测到 {prev_date}→{d} 跨日跳空 {gap*100:.1f}%，分钟线未复权，可能含除权除息影响"
                break
        prev_date = d
        prev_close = bar["close"]
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
    result = {
        "trades": [t.__dict__ for t in trades],
        "metrics": metrics,
        "nBars": len(kline),
        "period": cfg.period,
    }
    if div_warning:
        result["warning"] = div_warning
    return result


async def run_intraday_pool_backtest(codes: list[str], cfg: IntradayConfig) -> dict:
    """组合级分钟回测（P2-3）：多标的共享同一日内策略参数，各自独立进出场。

    返回每只股票的明细 + 组合汇总（总 PnL/笔数/胜率/per-trade Sharpe）。
    组合不调仓（各股独立全仓笔），聚焦"多标的日内策略批量评估"。
    """
    if not codes:
        return {"error": "codes 不能为空"}
    if len(codes) > 50:
        return {"error": "组合标的过多，单次最多 50 只"}

    per_code: list[dict] = []
    for code in codes:
        c = IntradayConfig(
            code=code, period=cfg.period, count=cfg.count,
            signal_lookback=cfg.signal_lookback, entry_threshold=cfg.entry_threshold,
            take_profit=cfg.take_profit, stop_loss=cfg.stop_loss,
            shares_per_trade=cfg.shares_per_trade, max_trades=cfg.max_trades,
            commissionRate=cfg.commissionRate, stampDuty=cfg.stampDuty,
            slippage=cfg.slippage, applyCost=cfg.applyCost,
        )
        try:
            res = await run_intraday_backtest(c)
            per_code.append({"code": code, **res})
        except Exception as e:
            per_code.append({"code": code, "error": str(e)})

    all_trades = [t for pc in per_code if "trades" in pc for t in pc["trades"]]
    returns = [t["pnl"] / (t["entry_price"] * t["shares"])
               for t in all_trades if t["entry_price"] * t["shares"] > 0]
    total_invest = sum(t["entry_price"] * t["shares"] for t in all_trades)
    # 分母用实际总投入资金，不做 0.95 臆造估算；保守口径避免高估收益
    peak_invest = total_invest
    cost_buy = (cfg.commissionRate + cfg.slippage) if cfg.applyCost else 0.0
    cost_sell = (cfg.commissionRate + cfg.stampDuty + cfg.slippage) if cfg.applyCost else 0.0
    metrics = {
        "nCodes": len(codes),
        "effectiveCodes": sum(1 for pc in per_code if "error" not in pc),
        "nTrades": len(all_trades),
        "winRate": sum(1 for r in returns if r > 0) / len(returns) if returns else 0.0,
        "totalPnl": sum(t["pnl"] for t in all_trades),
        "totalInvest": total_invest,
        "peakInvest": peak_invest,
        "totalReturn": (sum(t["pnl"] for t in all_trades) / peak_invest) if peak_invest > 0 else 0.0,
        "avgPnl": mean(returns) if returns else 0.0,
        "sharpe": sharpe_ratio(returns, periods_per_year=1) if returns else 0.0,
        "costRate": (cost_buy + cost_sell) if cfg.applyCost else 0.0,
        "applyCost": cfg.applyCost,
        "tPlus1": True,
    }
    return {"perCode": per_code, "metrics": metrics, "period": cfg.period, "pool": True}
