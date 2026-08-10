"""routers/artifacts.py 中间结果端点测试。

覆盖：保存/列表/读取/删除 全链路（联通流水线 API）。

artifacts 路由强制登录（Depends(require_user_id)），故测试 fixture 先注册
测试用户并在所有请求头带 Bearer token，避免 401。
"""
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import artifacts, db


@pytest.fixture
def client(tmp_path, monkeypatch):
    """隔离 artifact 目录 + 注册测试用户 + 注入 Authorization 头。

    返回的 client 已默认带登录头；需匿名访问时显式传 headers={} 覆盖。
    """
    monkeypatch.setattr(artifacts, "ARTIFACT_DIR", str(tmp_path))
    db.init_db()
    # 注册防刷按 client_ip 计数，TestClient 全程同一 IP 会累积；测试场景清空计数器
    from app import auth as _auth
    _auth._REGISTER_ATTEMPTS.clear()
    username = f"art_test_{uuid.uuid4().hex[:8]}"
    password = "pwd123456"
    with TestClient(app) as c:
        r = c.post("/api/auth/register", json={"username": username, "password": password})
        assert r.status_code == 200, r.text
        token = r.json()["token"]
        c.headers.update({"Authorization": f"Bearer {token}"})
        yield c


def test_save_list_get_delete(client):
    r = client.post("/api/artifacts", json={
        "kind": "select", "name": "测试选股",
        "payload": {"codes": ["sh600519"], "rows": []},
    })
    assert r.status_code == 200
    meta = r.json()
    assert meta["id"] and meta["kind"] == "select"

    r = client.get("/api/artifacts")
    assert r.status_code == 200
    assert any(a["id"] == meta["id"] for a in r.json())

    r = client.get(f"/api/artifacts/{meta['id']}")
    assert r.status_code == 200
    assert r.json()["payload"]["codes"] == ["sh600519"]

    r = client.delete(f"/api/artifacts/{meta['id']}")
    assert r.status_code == 200
    assert client.get(f"/api/artifacts/{meta['id']}").status_code == 404


def test_save_requires_payload(client):
    r = client.post("/api/artifacts", json={"kind": "select", "payload": {}})
    assert r.status_code == 400


def test_list_filter_by_kind(client):
    client.post("/api/artifacts", json={"kind": "select", "payload": {"a": 1}})
    client.post("/api/artifacts", json={"kind": "backtest", "payload": {"b": 2}})
    r = client.get("/api/artifacts?kind=select")
    assert r.status_code == 200
    assert all(a["kind"] == "select" for a in r.json())
    assert len(r.json()) == 1


def test_get_missing_404(client):
    assert client.get("/api/artifacts/no_such").status_code == 404
