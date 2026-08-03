"""intraday.py 组合级分钟回测单元测试（P2-3）。

覆盖：
- 多标的批量回测返回 perCode + 组合汇总 metrics
- 空 codes / 超限 codes 防御
- 单只拉取失败不影响其他标的
"""
import pytest

from app import intraday


def _make_minute_bars(n: int = 120, start: float = 10.0) -> list[dict]:
    """确定性分钟K线：单边上扬，日期跨两天（T+1 可平仓）。"""
    bars = []
    for i in range(n):
        date = "2024-03-01" if i < n // 2 else "2024-03-04"
        price = start * (1.001 ** i)
        bars.append({
            "datetime": f"{date} 10:{i % 60:02d}:00",
            "date": date, "open": price - 0.01, "close": price,
            "high": price + 0.02, "low": price - 0.03, "volume": 1000,
        })
    return bars


@pytest.mark.asyncio
async def test_pool_backtest_multi_code(monkeypatch):
    async def fake_minute(code, period, count):
        return _make_minute_bars(start=10.0 if code.endswith("1") else 20.0)
    monkeypatch.setattr("app.adapters.fetch_minute_kline", fake_minute)

    cfg = intraday.IntradayConfig(code="", period="5", count=120,
                                  signal_lookback=5, entry_threshold=0.001)
    res = await intraday.run_intraday_pool_backtest(["sh600001", "sz000002"], cfg)
    assert res["pool"] is True
    assert res["metrics"]["nCodes"] == 2
    assert res["metrics"]["effectiveCodes"] == 2
    assert len(res["perCode"]) == 2
    assert res["metrics"]["nTrades"] > 0
    assert res["metrics"]["totalPnl"] > 0  # 单边上扬 + 止盈策略必盈利
    assert "winRate" in res["metrics"]


@pytest.mark.asyncio
async def test_pool_backtest_one_code_fails(monkeypatch):
    """单只标的拉取失败应记录 error 而不整体崩溃。"""
    async def fake_minute(code, period, count):
        if code == "sh600001":
            raise Exception("network down")
        return _make_minute_bars()
    monkeypatch.setattr("app.adapters.fetch_minute_kline", fake_minute)

    cfg = intraday.IntradayConfig(code="", signal_lookback=5, entry_threshold=0.001)
    res = await intraday.run_intraday_pool_backtest(["sh600001", "sz000002"], cfg)
    assert res["metrics"]["nCodes"] == 2
    assert res["metrics"]["effectiveCodes"] == 1
    errors = [pc for pc in res["perCode"] if "error" in pc]
    assert len(errors) == 1
    assert errors[0]["code"] == "sh600001"


@pytest.mark.asyncio
async def test_pool_backtest_empty_codes():
    cfg = intraday.IntradayConfig(code="", signal_lookback=5)
    res = await intraday.run_intraday_pool_backtest([], cfg)
    assert "error" in res


@pytest.mark.asyncio
async def test_pool_backtest_too_many_codes():
    cfg = intraday.IntradayConfig(code="", signal_lookback=5)
    res = await intraday.run_intraday_pool_backtest([f"sh60{i:04d}" for i in range(60)], cfg)
    assert "error" in res
