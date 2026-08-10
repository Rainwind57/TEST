"""事件驱动回测引擎（P1-9）。

旧版仅「因子分层多空」+「ML 信号分层」两种范式，缺事件驱动范式——
突破/均值回归/套利的具体下单逻辑需用户自己映射到分层。本引擎补这一缺口：

- 逐 bar 推进，每日生成信号 → 下单 → 次日撮合（A 股 T+1）
- 涨跌停约束：涨停价买不进、跌停价卖不出（旧版分层回测也做跳过，此处更真实）
- 流动性约束：单股成交量上限 = 当日 volume * max_pct（默认 10%），超量部分成交
- 部分成交：超流动性上限的量递延次日继续撮合（旧版无此机制）
- 成本：佣金/印花税/滑点，与分层回测成本模型一致

与分层回测互补：分层用于因子评价，事件驱动用于策略验证（具体下单时序、容量约束）。
"""
import datetime
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import adapters
from .numpy_factors import kline_to_arrays, compute_factor_series, series_at
from .factors import round_trip_cost_rate


@dataclass
class Order:
    code: str
    side: str  # "buy" | "sell"
    qty: int
    order_type: str = "market"  # market | limit
    limit_price: float | None = None
    # 事件驱动独有：未成交部分递延次日继续（旧版分层回测无此机制）
    created_at: str = ""  # YYYY-MM-DD


@dataclass
class Fill:
    code: str
    side: str
    filled_qty: int
    avg_price: float
    slippage: float
    date: str


@dataclass
class Position:
    code: str
    qty: int = 0
    avg_cost: float = 0.0
    # T+1：记录买入批次 [(date, qty), ...]，避免 O(qty²)
    bought_dates: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class EventBacktestConfig:
    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.00025
    stamp_duty: float = 0.001
    slippage: float = 0.001
    apply_cost: bool = True
    max_volume_pct: float = 0.10  # 单股单日最多吃当日成交量 10%（流动性约束）
    max_position_pct: float = 0.20  # 单股权重上限 20%
    t_plus_1: bool = True


@dataclass
class EventBacktest:
    """事件驱动回测核心：逐 bar 推进，订单队列 + 撮合 + 持仓更新。"""
    config: EventBacktestConfig
    kline_by_code: dict[str, list[dict]]  # {code: [{date, open, close, high, low, volume}]}
    _cash: float = 0.0
    _positions: dict[str, Position] = field(default_factory=dict)
    _pending_orders: list[Order] = field(default_factory=list)
    _fills: list[Fill] = field(default_factory=list)
    _equity_curve: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self._cash = self.config.initial_cash
        # 预处理：每只股票转数组 + 缓存日期索引 + 涨跌停价
        self._arr_by_code = {}
        self._date_idx_by_code = {}
        self._limit_by_code = {}  # {code: {date: (upper, lower)}}
        for code, kline in self.kline_by_code.items():
            arr = kline_to_arrays(kline)
            self._arr_by_code[code] = arr
            self._date_idx_by_code[code] = {row["date"]: i for i, row in enumerate(kline)}
            # 涨跌停：A 股普通股 ±10%（ST ±5%，此处近似 10%）
            self._limit_by_code[code] = self._compute_limits(kline)
        self._sorted_dates_by_code = {c: sorted(d.keys()) for c, d in self._date_idx_by_code.items()}

    def _compute_limits(self, kline: list[dict]) -> dict[str, tuple[float, float]]:
        """逐日涨跌停价：基于前收盘 ±10%（A 股规则近似）。"""
        limits = {}
        prev_close = kline[0]["close"] if kline else 0
        for row in kline:
            upper = round(prev_close * 1.1, 2)
            lower = round(prev_close * 0.9, 2)
            limits[row["date"]] = (upper, lower)
            prev_close = row["close"]
        return limits

    def submit_order(self, order: Order):
        """提交订单到待撮合队列（次日撮合，T+1）。"""
        self._pending_orders.append(order)

    def run(self, signal_fn: Callable[[str, str, dict], list[Order]],
            dates: list[str] | None = None):
        """逐 bar 运行：每日生成信号 → 撮合昨日挂单 → 更新持仓 → 记净值。

        signal_fn(date, prev_date, state) -> list[Order]：用户自定义信号函数。
        state 含 cash 供决策。
        """
        if not self.kline_by_code:
            return self._result()
        ref_code = max(self.kline_by_code, key=lambda c: len(self.kline_by_code[c]))
        ref_dates = dates or [r["date"] for r in self.kline_by_code[ref_code]]
        prev_date = None
        for di, date in enumerate(ref_dates):
            is_last = (di == len(ref_dates) - 1)
            # 1. 撮合昨日挂单（T+1：今日撮合昨日提交的订单）；最后一日用收盘价撮合
            if self._pending_orders:
                self._match_orders(date, is_last_date=is_last)
            # 2. 生成今日信号 → 入挂单队列（明日撮合）
            # 最后一日不生成新信号：末日信号用当日收盘因子却撮合在当日开盘属反向前视
            if not is_last:
                state = {
                    "date": date, "prev_date": prev_date,
                    "cash": self._cash, "positions": {c: {"qty": p.qty, "avg_cost": p.avg_cost}
                                                       for c, p in self._positions.items()},
                    "pending_orders": [{"code": o.code, "side": o.side, "qty": o.qty}
                                       for o in self._pending_orders],
                    "market_data": self._market_snapshot(date),
                }
                new_orders = signal_fn(date, prev_date, state)
                for o in new_orders:
                    o.created_at = date
                    self.submit_order(o)
            # 3. 更新净值
            mv = self._market_value(date)
            self._equity_curve.append({"date": date, "cash": self._cash,
                                       "market_value": mv, "total": self._cash + mv})
            prev_date = date
        # 最后一日未成交的挂单强制撮合（以最后 close 成交）
        if self._pending_orders:
            last_date = ref_dates[-1] if ref_dates else ""
            self._match_orders(last_date, is_last_date=True)
            last_mv = self._market_value(last_date)
            if self._equity_curve:
                self._equity_curve[-1] = {"date": last_date, "cash": self._cash,
                                          "market_value": last_mv, "total": self._cash + last_mv}
        return self._result()

    def _market_snapshot(self, date: str) -> dict:
        """当日各股行情快照（open/close/volume/涨跌停）。"""
        snap = {}
        for code, arr in self._arr_by_code.items():
            idx = self._date_idx_by_code[code].get(date)
            if idx is None:
                continue
            limits = self._limit_by_code.get(code, {}).get(date, (None, None))
            snap[code] = {
                    "open": float(arr["open"][idx]) if "open" in arr else float(arr["close"][idx]),
                    "close": float(arr["close"][idx]),
                    "high": float(arr["high"][idx]),
                    "low": float(arr["low"][idx]),
                    "volume": float(arr["volume"][idx]),
                    "upper_limit": limits[0],
                    "lower_limit": limits[1],
                }
        return snap

    def _match_orders(self, date: str, is_last_date: bool = False):
        """撮合待挂单订单：涨跌停/流动性/部分成交。

        撮合价 = 当日 open（次日开盘成交，T+1 语义）；
        最后交易日用当日 close（避免末日信号用收盘因子成交在开盘的反向前视）。
        涨跌停递延（非撤单），最后交易日强制撮合。
        流动性：单股成交 <= volume * max_pct，超量递延。
        """
        snap = self._market_snapshot(date)
        still_pending = []
        for order in self._pending_orders:
            code = order.code
            s = snap.get(code)
            if not s or s["open"] is None or s["volume"] is None:
                still_pending.append(order)
                continue
            price = s["close"] if is_last_date else s["open"]
            upper, lower = s["upper_limit"], s["lower_limit"]
            # 涨跌停约束：递延而非撤单；最后交易日强制撮合
            if order.side == "buy" and upper is not None and price >= upper:
                if is_last_date:
                     pass  # 最后一天强制撮合
                else:
                    still_pending.append(order)
                    continue
            if order.side == "sell" and lower is not None and price <= lower:
                if is_last_date:
                    pass
                else:
                    still_pending.append(order)
                    continue
            # 滑点
            slip = price * self.config.slippage
            exec_price = price + slip if order.side == "buy" else price - slip
            # 流动性约束：单日最多吃 volume * max_pct
            max_vol = int(s["volume"] * self.config.max_volume_pct)
            filled_qty = min(order.qty, max(max_vol, 0))
            if filled_qty <= 0:
                still_pending.append(order)  # 无流动性，递延
                continue
            # T+1 卖出检查：只能卖 T 日前买入的量
            if order.side == "sell" and self.config.t_plus_1:
                pos = self._positions.get(code)
                if not pos:
                    continue
                sellable = sum(q for d, q in pos.bought_dates if d < date)
                filled_qty = min(filled_qty, sellable)
                if filled_qty <= 0:
                    continue
            # 部分成交：记录成交、扣减剩余量递延
            self._apply_fill(order, code, order.side, filled_qty, exec_price, date, slip)
            if filled_qty < order.qty:
                order.qty -= filled_qty
                still_pending.append(order)
        self._pending_orders = still_pending

    def _apply_fill(self, order: Order, code: str, side: str, qty: int,
                    price: float, date: str, slippage: float = 0.0):
        """应用成交：更新持仓/现金/记录。含现金校验与单股权重上限。

        关键：先校验所有约束（现金/持仓上限），仅在成交有效后才记录 fill，
        避免资金不足/超限时留下幽灵成交记录。
        """
        executed_qty = qty
        if side == "buy":
            per_share = price
            if self.config.apply_cost:
                per_share = price * (1 + self.config.commission_rate)
            max_affordable = int(self._cash / per_share) if per_share > 0 else 0
            if max_affordable <= 0:
                return
            executed_qty = min(executed_qty, max_affordable)
            total_assets = self._cash + self._market_value(date)
            max_position_qty = int(total_assets * self.config.max_position_pct / price) if price > 0 else 0
            pos = self._positions.get(code)
            current_qty = pos.qty if pos else 0
            executed_qty = min(executed_qty, max(0, max_position_qty - current_qty))
            if executed_qty <= 0:
                return
            amount = executed_qty * per_share
            self._cash -= amount
            pos = self._positions.setdefault(code, Position(code=code))
            new_qty = pos.qty + executed_qty
            pos.avg_cost = (pos.avg_cost * pos.qty + price * executed_qty) / new_qty if new_qty else 0
            pos.qty = new_qty
            pos.bought_dates.append((date, executed_qty))
        else:  # sell
            amount = executed_qty * price
            if self.config.apply_cost:
                amount *= (1 - self.config.commission_rate - self.config.stamp_duty)
            self._cash += amount
            pos = self._positions.get(code)
            if pos:
                pos.qty -= executed_qty
                remaining = executed_qty
                while remaining > 0 and pos.bought_dates:
                    batch_date, batch_qty = pos.bought_dates[0]
                    if batch_qty <= remaining:
                        pos.bought_dates.pop(0)
                        remaining -= batch_qty
                    else:
                        pos.bought_dates[0] = (batch_date, batch_qty - remaining)
                        remaining = 0
                if pos.qty <= 0:
                    del self._positions[code]
        self._fills.append(Fill(code=code, side=side, filled_qty=executed_qty,
                               avg_price=price, slippage=slippage, date=date))

    def _market_value(self, date: str) -> float:
        """持仓市值（用当日 close 估值；停牌日沿用最后有效收盘价）。"""
        mv = 0.0
        for code, pos in self._positions.items():
            closes = self._arr_by_code[code]["close"]
            # 停牌/无行情：二分查找日期之前最近的交易日
            if date not in self._date_idx_by_code[code]:
                dates_sorted = self._sorted_dates_by_code[code]
                import bisect
                i = bisect.bisect_left(dates_sorted, date) - 1
                if i >= 0:
                    idx = self._date_idx_by_code[code][dates_sorted[i]]
                    mv += float(closes[idx]) * pos.qty
                continue
            idx = self._date_idx_by_code[code][date]
            mv += float(closes[idx]) * pos.qty
        return mv

    def _result(self) -> dict:
        """回测结果：净值曲线 + 成交流水 + 持仓 + 绩效指标。"""
        from .factors import (annualized_return, annualized_volatility, sharpe_ratio,
                             max_drawdown, win_rate)
        equity = [e["total"] for e in self._equity_curve]
        rets = []
        for i in range(1, len(equity)):
            if equity[i - 1] > 0:
                rets.append(equity[i] / equity[i - 1] - 1)
        cum = equity[-1] / self.config.initial_cash - 1 if equity else 0
        return {
            "equity_curve": self._equity_curve,
            "fills": [{"code": f.code, "side": f.side, "qty": f.filled_qty,
                       "price": f.avg_price, "date": f.date} for f in self._fills],
            "final_positions": [{"code": c, "qty": p.qty, "avg_cost": p.avg_cost}
                                for c, p in self._positions.items()],
            "final_cash": self._cash,
            "metrics": {
                "cumulativeReturn": cum,
                "annualizedReturn": annualized_return(rets),
                "annualizedVolatility": annualized_volatility(rets),
                "sharpe": sharpe_ratio(rets),
                "maxDrawdown": max_drawdown(equity),
                "winRate": win_rate(rets),
                "fillCount": len(self._fills),
            },
        }
