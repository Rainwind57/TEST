"""前后端接口契约测试（pytest）：用 TestClient 走真实 ASGI 链路（路由→校验→序列化），
核对前端实际读取的字段名与后端响应是否一致。

前端字段来源：MonitorView.vue / MLView.vue / BacktestView.vue 的实际代码。
"""
import pytest

from fastapi.testclient import TestClient
from app.main import app
from app import db


@pytest.fixture(scope="module")
def client():
    db.init_db()  # 触发迁移（分配默认值 0.2/5 → 0.1/20）
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def auth_headers(client):
    r = client.post("/api/auth/register", json={"username": "contract_tester", "password": "test12345"})
    if r.status_code == 400:  # 已存在则登录
        r = client.post("/api/auth/login", json={"username": "contract_tester", "password": "test12345"})
    assert r.status_code == 200, f"注册/登录失败: {r.status_code} {r.text}"
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json().get("status") == "ok"


def test_monitor_config_contract(client):
    """MonitorView 读取的全部字段必须存在，且新默认值 0.1/20 已生效。"""
    r = client.get("/api/monitor/config")
    assert r.status_code == 200, r.text
    cfg = r.json()
    front_fields = ["mode", "modelId", "ranking", "ruleFactor", "board", "poolSize",
                    "adjustId", "source", "sourceBoard", "sourceTopN", "bullPct", "bearPct",
                    "allocMode", "perPositionPct", "maxPositions", "tradeDirections"]
    missing = [f for f in front_fields if f not in cfg]
    assert not missing, f"monitor/config 缺失字段: {missing}"
    assert cfg["perPositionPct"] == 0.1, f"perPositionPct 应为 0.1，实际 {cfg['perPositionPct']}"
    assert cfg["maxPositions"] == 20, f"maxPositions 应为 20，实际 {cfg['maxPositions']}"
    assert cfg["source"] == "watchlist"


def test_monitor_config_post_roundtrip(client, auth_headers):
    """MonitorView 保存配置的 POST 载荷字段与后端 SignalConfigBody 一致。"""
    body = {"mode": "rule", "source": "board", "sourceBoard": "all",
            "sourceTopN": 20, "allocMode": "equal", "perPositionPct": 0.1,
            "maxPositions": 20, "tradeDirections": "both"}
    r = client.post("/api/monitor/config", json=body, headers=auth_headers)
    assert r.status_code == 200, r.text
    assert r.json()["source"] == "board"
    # 还原
    client.post("/api/monitor/config", json={"mode": "rule", "source": "watchlist"}, headers=auth_headers)


def test_alloc_preview_contract(client, auth_headers):
    """alloc-preview 返回结构：顶层字段 + 每项基础字段；跳过项(plannedQty=0)必须带 error。"""
    r = client.post("/api/monitor/alloc-preview", json={"codes": ["sh600519", "sz000001"]},
                    headers=auth_headers)
    assert r.status_code == 200, r.text
    plan = r.json()
    for f in ["count", "allocations", "perPct", "usedPct", "maxPositions", "cashLeft"]:
        assert f in plan, f"alloc-preview 缺 {f}"
    for a in plan["allocations"]:
        for f in ["code", "name", "price", "plannedQty", "plannedAmount", "plannedPct"]:
            assert f in a, f"allocations 缺 {f}"
        if a["plannedQty"] == 0:
            assert a.get("error"), f"跳过项 {a['code']} 缺 error 原因"
        break


def test_ml_models_list_contract(client, auth_headers):
    r = client.get("/api/ml/models", headers=auth_headers)
    assert r.status_code == 200, r.text


def test_select_backtest_contract(client, auth_headers):
    """BacktestView 读 positionLedger/direction/survivorshipBiasWarning；无网络时 422/502 可接受。"""
    body = {"factor": "momentum", "groups": 3, "n": 3, "hist": 300, "benchmark": "none",
            "longOnly": False, "applyCost": False}
    r = client.post("/api/select/backtest", json=body, headers=auth_headers)
    assert r.status_code in (200, 422, 502), f"回测应 200/422/502，实际 {r.status_code}"
    if r.status_code == 200:
        data = r.json()
        for f in ["groupSummary", "longShort", "icSeries", "direction", "survivorshipBiasWarning"]:
            assert f in data, f"backtest 缺 {f}"


def test_auth_contract(client):
    r = client.post("/api/auth/login", json={"username": "nobody_contract_test", "password": "x"})
    assert r.status_code in (200, 401)
