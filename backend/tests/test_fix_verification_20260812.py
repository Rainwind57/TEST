"""修复验证测试：M1/M3/M5/W1 回归测试。

覆盖：
- M1: hist 自动抬升以覆盖 start_date（build_dataset + backtest_model）
- M3: reporting.py chartPos 无重复 id
- M5: _load_adjust 自动读取模型 featureWeights
- W1: scheduler 标的池来源扩展（watchlist/board/model_topn）
"""
import asyncio
import inspect
import json
import os
import tempfile

import numpy as np
import pytest

from app import ml, scheduler, db


# ============================================================
# M1: hist 自动抬升 —— build_dataset
# ============================================================

def _fake_klines(days=300, start_date="2024-01-01"):
    """构造确定性日期序列 K 线。"""
    from datetime import date, timedelta
    base = date.fromisoformat(start_date)
    out = []
    for i in range(days):
        d = base + timedelta(days=i)
        c = 10.0 * (1.001 ** i)
        out.append({
            "date": d.isoformat(),
            "open": c - 0.01, "close": c, "high": c + 0.02, "low": c - 0.03,
            "volume": 1000 + i * 10,
        })
    return out


def test_build_dataset_hist_auto_raise_for_past_period(monkeypatch):
    """M1: 指定历史 start_date 时 hist 自动抬升，确保 K 线覆盖时间段。"""
    from app import adapters

    # K 线只有 300 天（从 2024-01-01 开始），
    # 但用户传 hist=60 + start_date=2024-01-01 → 需要更多历史
    klines = _fake_klines(300, "2024-01-01")

    async def fake_market_list(board, limit, sort_field="amount"):
        return [{"code": "sh600000", "name": "测试股", "price": 10.0, "pctChg": 0.0,
                 "volume": 1e6, "amount": 1e7, "turnover": 1.0, "pe": 10.0, "pb": 1.0,
                 "mktCap": 100.0, "circMktCap": 80.0}]

    async def fake_kline(code, days):
        return klines[-days:] if len(klines) > days else klines

    monkeypatch.setattr(adapters, "fetch_market_list", fake_market_list)
    monkeypatch.setattr(adapters, "fetch_kline", fake_kline)
    monkeypatch.setattr(adapters, "fetch_market_list_multi",
                        lambda boards, limit: asyncio.ensure_future(
                            fake_market_list("all", limit)))

    # hist=60 但 start_date=2024-01-01 → 后端应自动抬升 hist
    ds = asyncio.run(ml.build_dataset("all", 10, 5, hist=60,
                                      start_date="2024-01-01", end_date="2024-06-30"))
    dates = ds["dates"]
    assert len(dates) > 0, "时间段内应有样本"
    assert all("2024-01-01" <= d <= "2024-06-30" for d in dates)


def test_build_dataset_hist_insufficient_error_message(monkeypatch):
    """M1: hist 不足以覆盖时间段时，错误信息应包含 hist 建议。"""
    from app import adapters

    klines = _fake_klines(200, "2025-01-01")

    async def fake_market_list(board, limit, sort_field="amount"):
        return [{"code": "sh600000", "name": "测试股", "price": 10.0, "pctChg": 0.0,
                 "volume": 1e6, "amount": 1e7, "turnover": 1.0, "pe": 10.0, "pb": 1.0,
                 "mktCap": 100.0, "circMktCap": 80.0}]

    async def fake_kline(code, days):
        return klines[-days:] if len(klines) > days else klines

    monkeypatch.setattr(adapters, "fetch_market_list", fake_market_list)
    monkeypatch.setattr(adapters, "fetch_kline", fake_kline)
    monkeypatch.setattr(adapters, "fetch_market_list_multi",
                        lambda boards, limit: asyncio.ensure_future(
                            fake_market_list("all", limit)))

    # 请求 2023 年数据，但 K 线从 2025 开始 → 应抛错且提示增大 hist
    with pytest.raises(ValueError, match="增大 hist"):
        asyncio.run(ml.build_dataset("all", 10, 5, hist=60,
                                     start_date="2023-01-01", end_date="2023-06-30"))


# ============================================================
# M3: reporting.py chartPos 无重复 id
# ============================================================

def test_reporting_chartpos_no_duplicate_id():
    """M3: reporting.py HTML 模板中 chartPos 不应有重复 id。"""
    from app import reporting
    src = inspect.getsource(reporting.render_html)
    # chartPos 在 id 中出现次数：外层 card + 内层 div → 修复后外层为 chartPosCard
    assert 'id="chartPosCard"' in src, "外层卡片应改为 chartPosCard"
    # 确保 id="chartPos" 只出现一次（在内层 echarts div）
    count = src.count('id="chartPos"')
    assert count == 1, f"chartPos id 应仅出现 1 次，实际 {count}"


# ============================================================
# M5: _load_adjust 自动读取模型 featureWeights
# ============================================================

def test_load_adjust_auto_reads_model_feature_weights(tmp_path, monkeypatch):
    """M5: 无显式 adjust 时，自动从模型侧车 JSON 读取 featureWeights。"""
    from app.routers import ml as ml_router

    # 创建假模型目录与侧车 JSON
    monkeypatch.setattr(ml, "ML_DIR", str(tmp_path))
    mid = "test_clone_model"
    meta = {
        "id": mid, "modelType": "gbdt",
        "featureNames": ["momentum", "rsi", "volatility"],
        "featureImportance": [
            {"feature": "momentum", "importance": 0.5},
            {"feature": "rsi", "importance": 0.3},
            {"feature": "volatility", "importance": 0.2},
        ],
        "featureWeights": {"momentum": 1.5, "rsi": 0.8, "volatility": 1.0},
        "threshold": 0.01,
    }
    with open(os.path.join(str(tmp_path), f"{mid}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    # 创建空的 joblib 占位（load_model_meta 不加载它，但检查路径存在）
    with open(os.path.join(str(tmp_path), f"{mid}.joblib"), "w") as f:
        f.write("")

    adjust = ml_router._load_adjust(None, None, modelId=mid)
    assert adjust is not None, "应自动读取到 featureWeights"
    assert adjust["featureWeights"] == {"momentum": 1.5, "rsi": 0.8, "volatility": 1.0}
    assert adjust["threshold"] == 0.01


def test_load_adjust_no_feature_weights_returns_none(tmp_path, monkeypatch):
    """M5: 模型无 featureWeights 时返回 None（不影响正常流程）。"""
    from app.routers import ml as ml_router

    monkeypatch.setattr(ml, "ML_DIR", str(tmp_path))
    mid = "test_plain_model"
    meta = {
        "id": mid, "modelType": "gbdt",
        "featureNames": ["momentum", "rsi"],
        "featureImportance": [
            {"feature": "momentum", "importance": 0.6},
            {"feature": "rsi", "importance": 0.4},
        ],
    }
    with open(os.path.join(str(tmp_path), f"{mid}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    with open(os.path.join(str(tmp_path), f"{mid}.joblib"), "w") as f:
        f.write("")

    adjust = ml_router._load_adjust(None, None, modelId=mid)
    assert adjust is None, "无 featureWeights 时应返回 None"


def test_load_adjust_explicit_overrides_auto(tmp_path, monkeypatch):
    """M5: 显式传 adjust 时优先使用（不读取模型自带）。"""
    from app.routers import ml as ml_router

    monkeypatch.setattr(ml, "ML_DIR", str(tmp_path))
    mid = "test_clone2"
    meta = {
        "id": mid, "featureNames": ["a", "b"],
        "featureImportance": [],
        "featureWeights": {"a": 2.0, "b": 3.0},
    }
    with open(os.path.join(str(tmp_path), f"{mid}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    with open(os.path.join(str(tmp_path), f"{mid}.joblib"), "w") as f:
        f.write("")

    explicit = {"featureWeights": {"a": 0.5}, "threshold": 0.1}
    adjust = ml_router._load_adjust(None, explicit, modelId=mid)
    assert adjust["featureWeights"] == {"a": 0.5}, "显式 adjust 应优先"


# ============================================================
# W1: scheduler 标的池来源扩展
# ============================================================

def test_scheduler_config_has_source_fields():
    """W1: get_signal_config 应包含 source/sourceBoard/sourceTopN 字段。"""
    cfg = scheduler.get_signal_config()
    assert "source" in cfg, "配置应包含 source 字段"
    assert "sourceBoard" in cfg, "配置应包含 sourceBoard 字段"
    assert "sourceTopN" in cfg, "配置应包含 sourceTopN 字段"
    assert cfg["source"] in ("watchlist", "board", "model_topn", ""), \
        f"source 值非法: {cfg['source']}"


def test_set_signal_config_accepts_source():
    """W1: set_signal_config 接受并持久化 source 字段。"""
    db.init_db()
    saved = scheduler.set_signal_config(
        "rule", source="board", source_board="gem", source_topn=30
    )
    assert saved["source"] == "board"
    assert saved["sourceBoard"] == "gem"
    assert saved["sourceTopN"] == 30

    # 恢复默认
    scheduler.set_signal_config("rule", source="watchlist")


def test_set_signal_config_rejects_invalid_source():
    """W1: 非法 source 值应抛出 ValueError。"""
    with pytest.raises(ValueError, match="source"):
        scheduler.set_signal_config("rule", source="invalid_source")


def test_set_signal_config_model_topn_requires_model_id():
    """W1: model_topn 来源必须指定 modelId。"""
    with pytest.raises(ValueError, match="model_topn"):
        scheduler.set_signal_config("rule", source="model_topn", model_id="")


# ============================================================
# 回归测试：已有功能不受影响
# ============================================================

def test_build_dataset_date_filter_still_works(monkeypatch):
    """回归：原有 start_date/end_date 过滤仍正常工作。"""
    from app import adapters

    klines = _fake_klines(300, "2024-01-01")

    async def fake_market_list(board, limit, sort_field="amount"):
        return [{"code": "sh600000", "name": "测试股", "price": 10.0, "pctChg": 0.0,
                 "volume": 1e6, "amount": 1e7, "turnover": 1.0, "pe": 10.0, "pb": 1.0,
                 "mktCap": 100.0, "circMktCap": 80.0}]

    async def fake_kline(code, days):
        return klines[-days:] if len(klines) > days else klines

    monkeypatch.setattr(adapters, "fetch_market_list", fake_market_list)
    monkeypatch.setattr(adapters, "fetch_kline", fake_kline)
    monkeypatch.setattr(adapters, "fetch_market_list_multi",
                        lambda boards, limit: asyncio.ensure_future(
                            fake_market_list("all", limit)))

    ds = asyncio.run(ml.build_dataset("all", 10, 5, 200,
                                      start_date="2024-03-01", end_date="2024-06-30"))
    dates = ds["dates"]
    assert len(dates) > 0
    assert all("2024-03-01" <= d <= "2024-06-30" for d in dates)


def test_min_hist_for_ml():
    """回归：min_hist_for_ml 返回合理值。"""
    m = ml.min_hist_for_ml(5)
    assert m > 60, "ML 需要足够历史覆盖长周期因子"
    assert m == ml._MAX_FACTOR_LOOKBACK + 5 + 1
