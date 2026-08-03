"""risk.py 路由自定义组合归因测试（联通：选股/组合优化结果 → 风险）。

覆盖：
- POST 指定 codes+weights 能完成归因（等权 & 传权重）
- 空 codes 防御
"""
import numpy as np
import pytest

from app.routers import risk as risk_router
from app import adapters


def _make_kline(n=80, start=10.0, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.standard_normal(n) * 0.02 + 0.0005
    closes = start * np.cumprod(1 + rets)
    return [
        {"date": f"2026-{i // 28 + 1:02d}-{i % 28 + 1:02d}",
         "open": float(c) * 0.99, "close": float(c),
         "high": float(c) * 1.02, "low": float(c) * 0.98, "volume": 100000}
        for i, c in enumerate(closes)
    ]


@pytest.fixture
def fake_market(monkeypatch):
    """mock 行情/K线：10 只股票，含全风格因子字段。"""
    codes = [f"sh60{i:04d}" for i in range(10)]

    async def fake_quotes(cs, source="tencent"):
        out = {}
        for i, c in enumerate(cs):
            out[c] = {
                "name": f"股{i}", "price": 10.0 + i,
                "preClose": 9.5 + i, "open": 9.6 + i,
                "turnover": 1.0 + i * 0.1, "pe": 12.0 + i, "pb": 1.5 + i * 0.1,
                "mktCap": 100.0 + i * 10, "circMktCap": 80.0 + i * 8,
            }
        return out

    async def fake_kline(code, days=150, force_refresh=False):
        return _make_kline(seed=int(code[-2:]))

    monkeypatch.setattr(adapters, "fetch_quotes", fake_quotes)
    monkeypatch.setattr(adapters, "fetch_kline", fake_kline)
    return codes


@pytest.mark.asyncio
async def test_attribution_custom_codes(fake_market):
    """10 只等权组合应完成 6 因子归因分解。"""
    res = await risk_router._run_attribution(fake_market)
    assert len(res["holdings"]) == 10
    assert "size" in res["factorNames"] and "beta" not in res["factorNames"]
    assert "exposures" in res
    assert "risk" in res and res["risk"]["totalRisk"] >= 0
    assert "var" in res


@pytest.mark.asyncio
async def test_attribution_with_weights(fake_market):
    """传权重时应按权重归一化（和=1）。"""
    n = len(fake_market)
    weights = [100.0 + i for i in range(n)]  # 非均匀权重
    res = await risk_router._run_attribution(fake_market, weights)
    total_w = sum(h["weight"] for h in res["holdings"])
    assert total_w == pytest.approx(1.0, abs=1e-9)
    assert res["holdings"][0]["weight"] != pytest.approx(res["holdings"][-1]["weight"])


@pytest.mark.asyncio
async def test_attribution_empty_codes():
    with pytest.raises(Exception):
        await risk_router._run_attribution([])


@pytest.mark.asyncio
async def test_attribution_too_few_codes(fake_market):
    """不足3只时应 422（样本不足防御）。"""
    import fastapi
    with pytest.raises(fastapi.HTTPException) as exc:
        await risk_router._run_attribution(fake_market[:2])
    assert exc.value.status_code == 422
