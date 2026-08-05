"""应用功能验证_机器学习与盯盘回测.md 修复回归测试。

覆盖：
- ML 默认/最小历史长度（hist 钳制，覆盖内置长周期因子）
- 候选池为空时区分「上游行情源故障」与「板块无匹配」（不再误报“样本不足”）
- ML 回测 K 线拉取带重试，失败计数进入错误信息
- 手动盯盘扫描接口与模型存在性校验
"""
import asyncio
import pytest

from app import ml, adapters


# ---------------- 历史长度钳制 ----------------

def test_min_hist_for_ml_covers_long_cycle_factors():
    """最小历史应覆盖内置最长因子（dist_52w 240 日窗口）+ 持有期。"""
    assert ml.min_hist_for_ml(5) >= 240 + 5 + 1
    assert ml.min_hist_for_ml(10) > ml.min_hist_for_ml(5)


def test_ml_default_hist_out_of_box():
    """训练/评估默认 hist 应 ≥ 最小要求，开箱即用不报“有效样本不足”。"""
    from app.routers import ml as ml_router
    assert ml_router.EvalBody().hist >= ml.min_hist_for_ml(5)
    assert ml_router.OptimizeMlBody().hist >= ml.min_hist_for_ml(5)


# ---------------- 候选池为空的可观测性 ----------------

async def _build_dataset_empty_pool(monkeypatch, degraded: bool):
    async def fake_market_list(board, limit, sort_field="amount"):
        return []

    async def fake_kline(code, days):
        return []

    monkeypatch.setattr(adapters, "fetch_market_list", fake_market_list)
    monkeypatch.setattr(adapters, "fetch_kline", fake_kline)
    monkeypatch.setattr(adapters, "_market_status",
                        {"degraded": degraded, "last_error": "sina: timeout", "ts": 1.0})
    with pytest.raises(ValueError) as exc:
        await ml.build_dataset("all", 100, 5, 240)
    return str(exc.value)


def test_build_dataset_empty_pool_reports_source_failure(monkeypatch):
    """上游行情源故障 → 报「候选池为空/上游失败」，而非误导性的“样本不足”。"""
    msg = asyncio.run(_build_dataset_empty_pool(monkeypatch, degraded=True))
    assert "候选池为空" in msg
    assert "上游行情列表获取失败" in msg


def test_build_dataset_empty_pool_no_degradation(monkeypatch):
    """未降级但池为空 → 提示板块无匹配股票。"""
    msg = asyncio.run(_build_dataset_empty_pool(monkeypatch, degraded=False))
    assert "候选池为空" in msg
    assert "无匹配股票" in msg


def _install_fake_model(tmp_path, monkeypatch):
    """在临时 ML_DIR 落一个最小模型 bundle，供打分/回测路径走到取池阶段。"""
    import joblib
    import numpy as np
    from sklearn.linear_model import LinearRegression
    monkeypatch.setattr(ml, "ML_DIR", str(tmp_path))
    model = LinearRegression()
    model.fit(np.random.default_rng(0).normal(size=(10, 3)),
              np.random.default_rng(1).normal(size=(10,)))
    joblib.dump({
        "model": model,
        "feature_names": ["momentum", "ma_dev", "volatility"],
        "model_type": "linear",
        "preprocess": {"lo": float("-inf"), "hi": float("inf"),
                       "mean": 0.0, "std": 1.0},
    }, tmp_path / "some_model.joblib")


def test_score_latest_empty_pool(monkeypatch, tmp_path):
    """打分遇空池应报「候选池为空」而非 NameError/无样本。"""
    _install_fake_model(tmp_path, monkeypatch)

    async def fake_market_list(board, limit, sort_field="amount"):
        return []

    monkeypatch.setattr(adapters, "fetch_market_list", fake_market_list)
    with pytest.raises(ValueError, match="候选池为空"):
        asyncio.run(ml.score_latest("some_model", "all", 100))


# ---------------- 历史钳制实际生效 ----------------

def _mk_kline(days: int, start: str = "2024-01-01"):
    out = []
    for i in range(days):
        c = 10.0 * (1.001 ** i)
        out.append({
            "date": f"2024-{1 + i // 28:02d}-{i % 28 + 1:02d}",
            "open": c - 0.01, "close": c, "high": c + 0.02, "low": c - 0.03,
            "volume": 1000 + i * 10,
        })
    return out


def test_build_dataset_clamps_small_hist(monkeypatch):
    """用户传 hist=240 时自动抬升到最小要求，内置长周期因子可用。"""
    klines = _mk_kline(280)

    async def fake_market_list(board, limit, sort_field="amount"):
        return [{"code": "sh600000", "name": "测试股", "price": 10.0, "pctChg": 0.0,
                 "volume": 1e6, "amount": 1e7, "turnover": 1.0, "pe": 10.0, "pb": 1.0,
                 "mktCap": 100.0, "circMktCap": 80.0}]

    async def fake_kline(code, days):
        assert days >= ml.min_hist_for_ml(5), "hist 应被钳制到最小要求以上"
        return klines

    monkeypatch.setattr(adapters, "fetch_market_list", fake_market_list)
    monkeypatch.setattr(adapters, "fetch_kline", fake_kline)

    ds = asyncio.run(ml.build_dataset("all", 10, 5, 240))
    assert len(ds["target"]) > 0


# ---------------- K 线重试 ----------------

def test_kline_retry_succeeds_after_failure():
    """首次抛错/返回空，重试后成功返回数据。"""
    calls = {"n": 0}

    async def flaky(code, days):
        calls["n"] += 1
        if calls["n"] < 3:
            return []
        return [{"date": "2024-01-01", "close": 10.0}]

    out = asyncio.run(ml._kline_retry(flaky, "sh600000", 100))
    assert len(out) == 1
    assert calls["n"] == 3


def test_kline_retry_returns_empty_after_all_failures():
    """持续失败最终返回 []（不抛异常，由调用方计数）。"""

    async def always_fail(code, days):
        raise RuntimeError("timeout")

    out = asyncio.run(ml._kline_retry(always_fail, "sh600000", 100))
    assert out == []


def test_backtest_model_empty_pool_message(monkeypatch, tmp_path):
    """ML 回测空池报「候选池为空」而非「K线≥40日仅 0 只」的误导信息。"""
    _install_fake_model(tmp_path, monkeypatch)

    async def fake_market_list(board, limit, sort_field="amount"):
        return []

    monkeypatch.setattr(adapters, "fetch_market_list", fake_market_list)
    with pytest.raises(ValueError, match="候选池为空"):
        asyncio.run(ml.backtest_model("some_model", "all", 60, 5, 5, 180))
