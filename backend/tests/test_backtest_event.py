"""backtest_event.py 事件驱动回测引擎单元测试（P1-9 + P2-13）。

覆盖：
- 引擎构造与涨跌停价计算
- 基本买卖成交 + 持仓/现金更新
- T+1 卖出约束（当日买次日才能卖）
- 涨跌停约束（涨停买不进、跌停卖不出）
- 流动性约束（超量部分递延）
- 部分成交
- 绩效指标输出
"""
import numpy as np
import pytest

from app import backtest_event as be


@pytest.fixture
def kline_uptrend():
    """单边上扬 60 根（无涨停，便于测试正常成交）。"""
    closes = [10.0 * (1.005 ** i) for i in range(60)]
    return [{"date": f"2024-01-{i//28+1:02d}-{i%28+1:02d}", "open": c, "close": c,
             "high": c*1.01, "low": c*0.99, "volume": 100000} for i, c in enumerate(closes)]


@pytest.fixture
def kline_limit_up():
    """构造涨停日：第 3 日从 10 涨到 11（+10%）。"""
    return [
        {"date": "2024-01-01", "open": 10, "close": 10, "high": 10, "low": 10, "volume": 50000},
        {"date": "2024-01-02", "open": 10, "close": 11, "high": 11, "low": 10, "volume": 50000},
        {"date": "2024-01-03", "open": 11, "close": 12.1, "high": 12.1, "low": 11, "volume": 50000},
    ]


def test_config_defaults():
    cfg = be.EventBacktestConfig()
    assert cfg.initial_cash == 1_000_000.0
    assert cfg.t_plus_1 is True
    assert cfg.max_volume_pct == 0.10


def test_engine_construction(kline_uptrend):
    bt = be.EventBacktest(be.EventBacktestConfig(), {"c0": kline_uptrend})
    assert "c0" in bt._arr_by_code
    assert "c0" in bt._date_idx_by_code
    assert "c0" in bt._limit_by_code


def test_limits_computed(kline_uptrend):
    """涨跌停价基于前收盘 ±10%。"""
    bt = be.EventBacktest(be.EventBacktestConfig(), {"c0": kline_uptrend})
    # 第一日无前收盘，用自身；第二日 upper = 第一日 close * 1.1
    limits = bt._limit_by_code["c0"]
    dates = [r["date"] for r in kline_uptrend]
    upper, lower = limits[dates[1]]
    assert upper == pytest.approx(kline_uptrend[0]["close"] * 1.1, rel=0.01)
    assert lower == pytest.approx(kline_uptrend[0]["close"] * 0.9, rel=0.01)


def test_basic_buy_and_sell(kline_uptrend):
    """买入成交 → 持仓增加、现金减少；卖出 → 反之。"""
    bt = be.EventBacktest(be.EventBacktestConfig(), {"c0": kline_uptrend})
    bt.submit_order(be.Order(code="c0", side="buy", qty=100))
    # 次日撮合（T+1）
    dates = [r["date"] for r in kline_uptrend]
    bt._match_orders(dates[1])
    assert bt._positions["c0"].qty == 100
    assert bt._cash < be.EventBacktestConfig().initial_cash
    # 卖出（需 T+1，第 3 日才能卖第 1 日买的）
    bt.submit_order(be.Order(code="c0", side="sell", qty=100))
    bt._match_orders(dates[2])
    assert "c0" not in bt._positions or bt._positions["c0"].qty == 0


def test_t_plus_1_sell_blocked(kline_uptrend):
    """T+1：当日买当日不能卖。"""
    cfg = be.EventBacktestConfig()
    bt = be.EventBacktest(cfg, {"c0": kline_uptrend})
    dates = [r["date"] for r in kline_uptrend]
    # 第 1 日买
    bt.submit_order(be.Order(code="c0", side="buy", qty=100))
    bt._match_orders(dates[1])
    assert bt._positions["c0"].qty == 100
    # 第 1 日（同日）尝试卖 → 应被 T+1 拦截
    bt.submit_order(be.Order(code="c0", side="sell", qty=100))
    bt._match_orders(dates[1])  # 同日撮合，bought_dates 全 = 今日，sellable=0
    assert bt._positions["c0"].qty == 100  # 未卖出


def test_limit_up_blocks_buy():
    """涨停日买单被撤（买不进）。

    构造：第 1 日 close=10，第 2 日 open=11（=10*1.1 涨停价）→ 买单应被跳过。
    """
    kline = [
        {"date": "2024-01-01", "open": 10, "close": 10, "high": 10, "low": 10, "volume": 50000},
        {"date": "2024-01-02", "open": 11, "close": 11, "high": 11, "low": 11, "volume": 50000},
        {"date": "2024-01-03", "open": 11, "close": 11, "high": 11, "low": 11, "volume": 50000},
    ]
    bt = be.EventBacktest(be.EventBacktestConfig(), {"c0": kline})
    dates = [r["date"] for r in kline]
    # 第 2 日 open=11 = upper(10*1.1=11) → 涨停，买单应被撤
    bt.submit_order(be.Order(code="c0", side="buy", qty=100))
    bt._match_orders(dates[1])
    assert "c0" not in bt._positions  # 买不进，无持仓
    assert len(bt._fills) == 0


def test_liquidity_constraint():
    """流动性约束：单日最多吃 volume*10%，超量递延。"""
    kline = [{"date": f"2024-01-{i+1:02d}", "open": 10, "close": 10,
              "high": 10, "low": 10, "volume": 1000} for i in range(5)]
    cfg = be.EventBacktestConfig(max_volume_pct=0.10)
    bt = be.EventBacktest(cfg, {"c0": kline})
    # 下单 200 股，但 max_vol = 1000*0.1 = 100 → 部分成交 100，剩 100 递延
    bt.submit_order(be.Order(code="c0", side="buy", qty=200))
    dates = [r["date"] for r in kline]
    bt._match_orders(dates[1])
    assert bt._positions["c0"].qty == 100  # 仅成交 100
    assert len(bt._pending_orders) == 1  # 剩余递延
    # 次日继续撮合递延单
    bt._match_orders(dates[2])
    assert bt._positions["c0"].qty == 200  # 全部成交


def test_partial_fill_slippage(kline_uptrend):
    """滑点：买入价 = open + slippage。"""
    cfg = be.EventBacktestConfig(slippage=0.001, apply_cost=False)
    bt = be.EventBacktest(cfg, {"c0": kline_uptrend})
    bt.submit_order(be.Order(code="c0", side="buy", qty=100))
    dates = [r["date"] for r in kline_uptrend]
    bt._match_orders(dates[1])
    fill = bt._fills[-1]
    expected_price = kline_uptrend[1]["close"] * (1 + 0.001)
    assert fill.avg_price == pytest.approx(expected_price, rel=0.001)


def test_run_end_to_end(kline_uptrend):
    """端到端：跑完整个回测，出净值曲线 + 成交 + 指标。"""
    bt = be.EventBacktest(be.EventBacktestConfig(), {"c0": kline_uptrend})

    def sig(date, prev, state):
        return [be.Order(code="c0", side="buy", qty=100)] if prev is None else []

    r = bt.run(sig)
    assert len(r["equity_curve"]) == len(kline_uptrend)
    assert len(r["fills"]) >= 1
    assert "metrics" in r
    assert "cumulativeReturn" in r["metrics"]
    assert "sharpe" in r["metrics"]


def test_run_empty_kline():
    """空 K 线不崩。"""
    bt = be.EventBacktest(be.EventBacktestConfig(), {})
    r = bt.run(lambda d, p, s: [])
    assert r["equity_curve"] == []
    assert r["fills"] == []


def test_market_snapshot(kline_uptrend):
    bt = be.EventBacktest(be.EventBacktestConfig(), {"c0": kline_uptrend})
    dates = [r["date"] for r in kline_uptrend]
    snap = bt._market_snapshot(dates[1])
    assert "c0" in snap
    assert snap["c0"]["close"] == kline_uptrend[1]["close"]
    assert snap["c0"]["upper_limit"] is not None
