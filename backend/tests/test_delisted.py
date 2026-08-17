"""退市股清单（生存偏差修复）单元测试。"""
import asyncio
import json
import os

import pytest

from app import adapters


class _FakeResp:
    def __init__(self, payload: dict | None = None, content: bytes = b""):
        self._payload = payload
        self.content = content

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def test_sse_delisted_parses_original_codes():
    """_fetch_sse_delisted 应解析出原 A 股代码 + 上市/退市日期，并跳过未退市股。"""
    payload = {
        "result": [
            {"A_STOCK_CODE": "600001", "COMPANY_ABBR": "邯郸钢铁",
             "LIST_DATE": "19980122", "DELIST_DATE": "20091229"},
            {"A_STOCK_CODE": "600000", "COMPANY_ABBR": "浦发银行",
             "LIST_DATE": "19991110", "DELIST_DATE": "-"},
            {"A_STOCK_CODE": "600002", "COMPANY_ABBR": "齐鲁退市",
             "LIST_DATE": "19980408", "DELIST_DATE": "20060424"},
        ]
    }

    async def fake_get(url, timeout, headers=None):
        return _FakeResp(payload=payload)

    original = adapters._http_get
    adapters._http_get = fake_get
    try:
        rows = asyncio.run(adapters._fetch_sse_delisted())
    finally:
        adapters._http_get = original

    assert len(rows) == 2
    assert rows[0] == {"code": "sh600001", "name": "邯郸钢铁",
                       "list_date": "19980122", "delist_date": "20091229"}
    assert rows[1]["code"] == "sh600002"


def test_delisted_cache_file_roundtrip(tmp_path, monkeypatch):
    """fetch_delisted_stocks 应读取持久缓存（命中时不再请求网络）。"""
    fake = [{"code": "sh600001", "name": "邯郸钢铁",
             "list_date": "19980122", "delist_date": "20091229"}]
    cache_file = tmp_path / "delisted_stocks.json"
    cache_file.write_text(json.dumps(fake, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(adapters, "_DELISTED_CACHE_FILE", str(cache_file))
    monkeypatch.setattr(adapters, "_DELISTED_CACHE", {})

    called = {"n": 0}

    async def boom(*a, **k):
        called["n"] += 1
        raise RuntimeError("network should not be called on cache hit")

    monkeypatch.setattr(adapters, "_fetch_sse_delisted", boom)
    monkeypatch.setattr(adapters, "_fetch_szse_delisted", boom)

    rows = asyncio.run(adapters.fetch_delisted_stocks())
    assert rows == fake
    assert called["n"] == 0


def test_delisted_cache_file_missing_goes_network(tmp_path, monkeypatch):
    """缓存文件不存在时应回源网络（网络失败则返回空，不抛异常）。"""
    monkeypatch.setattr(adapters, "_DELISTED_CACHE_FILE", str(tmp_path / "nope.json"))
    monkeypatch.setattr(adapters, "_DELISTED_CACHE", {})

    async def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(adapters, "_fetch_sse_delisted", boom)
    monkeypatch.setattr(adapters, "_fetch_szse_delisted", boom)

    rows = asyncio.run(adapters.fetch_delisted_stocks())
    assert rows == []
