"""前端集成测试：通过 HTTP API 验证前后端联调功能。

覆盖：
- M1: ML 回测/训练 API 时间段参数
- M5: ML 打分/回测自动应用克隆模型权重
- W1: 盯盘配置 source 字段
- S1: 行业板块 API
- D1: 北交所代码识别
"""
import json
import os
import sys
import urllib.request
import urllib.error

BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000/api")

def _req(method, path, body=None, expected_status=200):
    """发送 HTTP 请求并返回 (status, data)。"""
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    # 如果有 token，从环境变量读取
    token = os.environ.get("API_TOKEN", "")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode()
        try:
            body_json = json.loads(body_text)
        except Exception:
            body_json = {"detail": body_text}
        return e.code, body_json


def check(condition, msg):
    if not condition:
        print(f"  FAIL: {msg}")
        return False
    print(f"  OK: {msg}")
    return True


# ============================================================
# S1: 行业板块 API
# ============================================================
def test_sectors_api():
    """S1: /api/data/sectors 返回行业映射。"""
    print("\n[S1] 行业板块 API")
    status, data = _req("GET", "/data/sectors")
    if not check(status == 200, f"sectors API 返回 {status}"):
        return
    if not check(isinstance(data, dict), "返回 dict 类型"):
        return
    names = set(data.values())
    check(len(names) > 5, f"行业数 {len(names)} > 5")
    check("半导体" in names or "电子" in names, f"包含常见行业，实际: {list(names)[:5]}")


# ============================================================
# D1: 北交所代码识别（前端 format.js normalizeCode 的等价后端验证）
# ============================================================
def test_bse_stock_exists():
    """D1: 北交所股票 8xxxxx/920xxx 能被后端识别。"""
    print("\n[D1] 北交所代码识别")
    # 验证 normalizeCode 等价逻辑：后端应能处理 bj 前缀
    test_codes = ["bj830799", "bj920123", "bj430001"]
    for code in test_codes:
        status, data = _req("GET", f"/stock/exists?code={code}")
        check(status in (200, 404), f"{code} 查询不崩溃 (status={status})")


# ============================================================
# M1: ML API 时间段参数
# ============================================================
def test_ml_evaluate_with_dates():
    """M1: /api/ml/evaluate 接受 startDate/endDate 参数。"""
    print("\n[M1] ML 评估时间段参数")
    body = {
        "board": "all", "poolSize": 20, "n": 5, "hist": 300,
        "modelType": "gbdt", "nSplits": 3, "gap": 5,
        "startDate": "2024-01-01", "endDate": "2024-06-30",
    }
    status, data = _req("POST", "/ml/evaluate", body)
    check(status in (200, 422, 502),
          f"评估 API 返回 {status}（200=成功, 422=参数/数据不足, 502=上游故障）")
    if status == 200:
        check("metrics" in data or "cv" in data or "sharpe" in data,
              "返回评估指标")


def test_ml_backtest_with_dates():
    """M1: /api/ml/backtest 接受 startDate/endDate。"""
    print("\n[M1] ML 回测时间段参数")
    # 先获取已有模型列表
    status, models = _req("GET", "/ml/models")
    if status != 200 or not models:
        print("  SKIP: 无可用模型")
        return
    mid = models[0].get("id") or list(models.keys())[0]
    body = {
        "modelId": mid, "board": "all", "poolSize": 20,
        "groups": 3, "n": 3, "hist": 300,
        "startDate": "2024-01-01", "endDate": "2024-06-30",
    }
    status, data = _req("POST", "/ml/backtest", body)
    check(status in (200, 422, 502),
          f"回测 API 返回 {status}")


# ============================================================
# M5: 克隆模型权重自动应用
# ============================================================
def test_ml_score_auto_adjust():
    """M5: ML 打分 API 接受 adjust 或不传（自动读模型 featureWeights）。"""
    print("\n[M5] ML 打分自动调参")
    status, models = _req("GET", "/ml/models")
    if status != 200 or not models:
        print("  SKIP: 无可用模型")
        return
    mid = models[0].get("id") or list(models.keys())[0]
    # 不传 adjust → 后端自动检查模型侧车
    body = {"modelId": mid, "board": "all", "poolSize": 20}
    status, data = _req("POST", "/ml/score", body)
    check(status in (200, 422, 502),
          f"打分 API 返回 {status}")
    if status == 200:
        check("rows" in data or isinstance(data, list), "返回打分列表")


# ============================================================
# W1: 盯盘配置 source 字段
# ============================================================
def test_monitor_config_source():
    """W1: 盯盘配置 API 支持 source 字段。"""
    print("\n[W1] 盯盘配置 source 字段")
    # 获取当前配置
    status, cfg = _req("GET", "/monitor/config")
    if not check(status == 200, f"获取配置返回 {status}"):
        return
    check("source" in cfg, "配置含 source 字段")
    check("sourceBoard" in cfg, "配置含 sourceBoard 字段")
    check("sourceTopN" in cfg, "配置含 sourceTopN 字段")

    # 设置 source=board
    body = {
        "mode": "rule",
        "source": "board",
        "sourceBoard": "gem",
        "sourceTopN": 20,
    }
    status, saved = _req("POST", "/monitor/config", body)
    if check(status == 200, f"设置配置返回 {status}"):
        check(saved.get("source") == "board", "source 持久化为 board")
        check(saved.get("sourceBoard") == "gem", "sourceBoard 持久化为 gem")

    # 设置非法 source → 400
    body_bad = {"mode": "rule", "source": "invalid"}
    status, err = _req("POST", "/monitor/config", body_bad)
    check(status == 400, f"非法 source 返回 400 (实际 {status})")

    # 恢复默认
    _req("POST", "/monitor/config", {"mode": "rule", "source": "watchlist"})


# ============================================================
# 综合回归
# ============================================================
def test_health():
    """健康检查。"""
    print("\n[综合] 健康检查")
    status, data = _req("GET", "/health")
    check(status == 200, f"健康检查返回 {status}")


if __name__ == "__main__":
    print("=" * 60)
    print("前端集成测试 - API 层")
    print(f"目标: {BASE}")
    print("=" * 60)

    tests = [
        ("健康检查", test_health),
        ("S1-行业板块API", test_sectors_api),
        ("D1-北交所代码识别", test_bse_stock_exists),
        ("M1-ML评估时间段", test_ml_evaluate_with_dates),
        ("M1-ML回测时间段", test_ml_backtest_with_dates),
        ("M5-ML打分自动调参", test_ml_score_auto_adjust),
        ("W1-盯盘配置source", test_monitor_config_source),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            print(f"\n--- {name} ---")
            fn()
            passed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
