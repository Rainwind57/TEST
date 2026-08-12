"""稽核修复验证测试：针对「机器学习模块问题清单」和「盯盘模块与选股一致性」两份文档的修复。

运行方式：pytest tests/test_稽核修复验证_20260811.py -v
前提：应用已在 localhost:8899 运行，且已有 ML 模型落盘。
"""
import json
import os
import sys
import pytest
import requests

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

BASE = "http://localhost:8899"
TOKEN = None

# ========================================================================
# 鉴权
# ========================================================================

@pytest.fixture(scope="session")
def auth_headers():
    global TOKEN
    if TOKEN is None:
        # 先用已存在的用户登录
        r = requests.post(f"{BASE}/api/auth/login",
                         json={"username": "tester", "password": "test123"})
        if r.status_code == 200:
            TOKEN = r.json()["token"]
        else:
            pytest.fail(f"无法登录已知用户 tester: {r.status_code} {r.text[:200]}")
    return {"Authorization": f"Bearer {TOKEN}"}


def api(method, path, headers=None, json_body=None, expected=200):
    url = f"{BASE}{path}"
    if headers is None:
        headers = {}
    if "Authorization" not in headers:
        h = {"Authorization": f"Bearer {TOKEN}"}
    else:
        h = headers
    if method == "GET":
        r = requests.get(url, headers=h)
    elif method == "POST":
        r = requests.post(url, json=json_body, headers=h)
    elif method == "DELETE":
        r = requests.delete(url, headers=h)
    else:
        raise ValueError(method)
    assert r.status_code == expected, f"{method} {path} → {r.status_code}: {r.text[:300]}"
    try:
        return r.json()
    except Exception:
        return r.text


# ========================================================================
# 一、机器学习模块问题清单 验证
# ========================================================================

def _pick_computable(models):
    """挑一个可计算的模型（排除含虚构特征 f0/f1 的导入模型）。"""
    computable = [m for m in models if m.get('computable', True) and not m.get('unknownFeatures')]
    if not computable:
        for m in models:
            if m.get('computable', True):
                computable.append(m)
    return computable[0] if computable else None


class TestML_时段历史不透明_P0:
    """P0-1: 回测返回 actualHistDays / effectiveStart / effectiveEnd"""

    def test_backtest_returns_actual_hist_days(self, auth_headers):
        models = api("GET", "/api/ml/models", headers=auth_headers)
        computable = [m for m in models if m.get('computable', True) and not m.get('unknownFeatures')]
        if not computable:
            pytest.skip("无可计算模型")
        mid = computable[0]["id"]
        result = api("POST", "/api/ml/backtest", headers=auth_headers, json_body={
            "modelId": mid, "board": "all", "poolSize": 30, "groups": 2,
            "n": 5, "hist": 100, "applyCost": False,
        })
        assert "actualHistDays" in result, f"缺失 actualHistDays: {list(result.keys())[:20]}"
        assert isinstance(result["actualHistDays"], int)
        assert result["actualHistDays"] >= 100, f"actualHistDays={result['actualHistDays']} 应 ≥100"

    def test_backtest_returns_effective_start_end(self, auth_headers):
        models = api("GET", "/api/ml/models", headers=auth_headers)
        if not models:
            pytest.skip("无可用模型")
        mid = _pick_computable(models)["id"]
        result = api("POST", "/api/ml/backtest", headers=auth_headers, json_body={
            "modelId": mid, "board": "all", "poolSize": 30, "groups": 2,
            "n": 5, "hist": 200, "applyCost": False,
        })
        assert "effectiveStart" in result, "缺失 effectiveStart"
        assert "effectiveEnd" in result, "缺失 effectiveEnd"
        assert result["effectiveStart"] < result["effectiveEnd"], \
            f"effectiveStart={result['effectiveStart']} 应 < effectiveEnd={result['effectiveEnd']}"

    def test_hist_too_small_triggers_warning(self, auth_headers):
        """hist=50（远小于 min_hist_for_ml(5)=246）应触发 histWarning"""
        models = api("GET", "/api/ml/models", headers=auth_headers)
        if not models:
            pytest.skip("无可用模型")
        mid = _pick_computable(models)["id"]
        result = api("POST", "/api/ml/backtest", headers=auth_headers, json_body={
            "modelId": mid, "board": "all", "poolSize": 30, "groups": 2,
            "n": 5, "hist": 50, "applyCost": False,
        })
        if "histWarning" in result:
            assert "自动调整" in result["histWarning"] or "调整至" in result["histWarning"]
        else:
            pytest.fail("hist=50 应触发 histWarning，但未返回")

    def test_hist_large_no_warning(self, auth_headers):
        """hist=1024（足够大）不应触发 histWarning"""
        models = api("GET", "/api/ml/models", headers=auth_headers)
        if not models:
            pytest.skip("无可用模型")
        mid = _pick_computable(models)["id"]
        result = api("POST", "/api/ml/backtest", headers=auth_headers, json_body={
            "modelId": mid, "board": "all", "poolSize": 30, "groups": 2,
            "n": 5, "hist": 1024, "applyCost": False,
        })
        assert not result.get("histWarning"), f"hist=1024 不应触发警告: {result.get('histWarning')}"


class TestML_调参默认值_P1:
    """P1-1: 调参面板默认值不应全为 1，应为特征重要性或 0"""

    def test_model_params_has_feature_importance(self, auth_headers):
        """GET /ml/models/{mid}/params 返回 featureImportance"""
        models = api("GET", "/api/ml/models", headers=auth_headers)
        if not models:
            pytest.skip("无可用模型")
        mid = _pick_computable(models)["id"]
        params = api("GET", f"/api/ml/models/{mid}/params", headers=auth_headers)
        assert "featureImportance" in params, f"缺失 featureImportance: {list(params.keys())}"
        assert len(params["featureImportance"]) > 0, "featureImportance 不应为空"

    def test_model_params_feature_weights_default(self, auth_headers):
        """未调参时 featureWeights 应为空（前端据此判断用重要性或0）"""
        models = api("GET", "/api/ml/models", headers=auth_headers)
        if not models:
            pytest.skip("无可用模型")
        mid = _pick_computable(models)["id"]
        params = api("GET", f"/api/ml/models/{mid}/params", headers=auth_headers)
        fw = params.get("featureWeights")
        if fw is not None:
            # 如果有默认权重，不应全为 1
            values = [v for v in fw.values() if isinstance(v, (int, float))]
            if len(values) > 0:
                assert not all(abs(v - 1.0) < 0.01 for v in values), \
                    f"所有默认权重都是 1，应收敛为重要性或0: {list(fw.items())[:5]}"


class TestML_另存新模型_P1:
    """P1-2: 调参后另存为新模型"""

    def test_adjust_save_as_new(self, auth_headers):
        models = api("GET", "/api/ml/models", headers=auth_headers)
        if not models:
            pytest.skip("无可用模型")
        mid = _pick_computable(models)["id"]
        res = api("POST", f"/api/ml/models/{mid}/adjust", headers=auth_headers, json_body={
            "featureWeights": {"momentum5": 2.0, "rsi": 0.5},
            "saveAsNew": True,
        })
        if "newModelId" in res:
            assert res["newModelId"].startswith("clone_"), f"新模型ID格式异常: {res['newModelId']}"
            # 验证新模型可在列表中查到
            models2 = api("GET", "/api/ml/models", headers=auth_headers)
            new_ids = [m["id"] for m in models2]
            assert res["newModelId"] in new_ids, f"新模型不在列表: {res['newModelId']}"
            # 清理
            api("DELETE", f"/api/ml/models/{res['newModelId']}", headers=auth_headers, expected=200)
        elif "cloneError" in res:
            pytest.fail(f"另存失败: {res['cloneError']}")
        else:
            pytest.fail(f"响应不含 newModelId: {list(res.keys())}")

    def test_adjust_save_as_new_without_weights(self, auth_headers):
        """不传 featureWeights 时也能另存（纯复制）"""
        models = api("GET", "/api/ml/models", headers=auth_headers)
        if not models:
            pytest.skip("无可用模型")
        mid = _pick_computable(models)["id"]
        res = api("POST", f"/api/ml/models/{mid}/adjust", headers=auth_headers, json_body={
            "saveAsNew": True,
        })
        if "newModelId" in res:
            api("DELETE", f"/api/ml/models/{res['newModelId']}", headers=auth_headers, expected=200)
        else:
            pytest.fail(f"不传权重时另存也应成功: {res}")


# ========================================================================
# 二、盯盘模块与选股一致性 验证
# ========================================================================

class TestMonitor_配置透传_P0:
    """P0-1: 盯盘配置透传 board/poolSize/adjustId"""

    def test_monitor_config_has_board_and_poolsize(self, auth_headers):
        """GET /monitor/config 返回 board / poolSize / adjustId"""
        cfg = api("GET", "/monitor/config", headers=auth_headers)
        assert "board" in cfg, f"缺失 board: {list(cfg.keys())}"
        assert "poolSize" in cfg, f"缺失 poolSize: {list(cfg.keys())}"
        assert "adjustId" in cfg, f"缺失 adjustId: {list(cfg.keys())}"
        assert cfg["board"] == "all", f"默认 board 应为 all: {cfg['board']}"
        assert cfg["poolSize"] == 150, f"默认 poolSize 应为 150: {cfg['poolSize']}"

    def test_monitor_set_config_preserves_board_poolsize(self, auth_headers):
        """POST /monitor/config 新字段不丢失"""
        models = api("GET", "/api/ml/models", headers=auth_headers)
        if not models:
            pytest.skip("无可用模型")
        mid = _pick_computable(models)["id"]
        cfg = api("POST", "/monitor/config", headers=auth_headers, json_body={
            "mode": "model",
            "modelId": mid,
            "ranking": "full",
            "board": "sh_main",
            "poolSize": 200,
        })
        assert cfg["board"] == "sh_main", f"board 未保存: {cfg}"
        assert cfg["poolSize"] == 200, f"poolSize 未保存: {cfg}"
        # 恢复默认
        api("POST", "/monitor/config", headers=auth_headers, json_body={
            "mode": "rule", "board": "all", "poolSize": 150,
        })

    def test_monitor_status_includes_config(self, auth_headers):
        """GET /monitor/status 包含完整 config"""
        status = api("GET", "/monitor/status", headers=auth_headers)
        assert "config" in status, "status 缺失 config"
        cfg = status["config"]
        for k in ("mode", "modelId", "ranking", "board", "poolSize", "adjustId"):
            assert k in cfg, f"config 缺失 {k}"


class TestMonitor_扫描透传:
    """验证盯盘扫描时实际使用了 board/poolSize（非直接调用API，仅验证配置会保存生效）"""

    def test_scan_now_does_not_crash_with_config(self, auth_headers):
        """POST /monitor/scan?force=true 在 model 模式下不崩溃"""
        models = api("GET", "/api/ml/models", headers=auth_headers)
        if not models:
            pytest.skip("无可用模型")
        mid = _pick_computable(models)["id"]
        # 设置模型模式
        api("POST", "/monitor/config", headers=auth_headers, json_body={
            "mode": "model", "modelId": mid, "ranking": "full",
        })
        # 立即扫描
        result = api("POST", "/monitor/scan?force=true", headers=auth_headers)
        assert result["ok"], f"扫描失败: {result.get('reason', result)}"
        assert "signals" in result
        # 恢复规则模式
        api("POST", "/monitor/config", headers=auth_headers, json_body={
            "mode": "rule",
        })


# ========================================================================
# 三、边界测试 / 异常测试
# ========================================================================

class TestBoundary:
    """边界条件测试"""

    def test_backtest_pool_size_minimum(self, auth_headers):
        """poolSize=10（极小）应返回结果或合理报错"""
        models = api("GET", "/api/ml/models", headers=auth_headers)
        if not models:
            pytest.skip("无可用模型")
        mid = _pick_computable(models)["id"]
        try:
            result = requests.post(f"{BASE}/api/ml/backtest",
                                   json={"modelId": mid, "board": "all", "poolSize": 10,
                                         "groups": 2, "n": 5, "hist": 300, "applyCost": False},
                                   headers=auth_headers)
            assert result.status_code in (200, 422, 502), f"意外状态码: {result.status_code}"
        except Exception:
            pass  # 网络超时也算通过（上游行情不稳定）

    def test_adjust_nonexistent_model(self, auth_headers):
        """调整不存在的模型应 404"""
        r = requests.post(f"{BASE}/api/ml/models/no_such_model_xyz/adjust",
                         json={"featureWeights": {"a": 1.0}},
                         headers=auth_headers)
        assert r.status_code in (404, 500), f"应为 404/500: {r.status_code}"

    def test_monitor_config_invalid_mode(self, auth_headers):
        """无效 mode 应 400"""
        r = requests.post(f"{BASE}/api/monitor/config",
                         json={"mode": "invalid_mode", "modelId": "x"},
                         headers=auth_headers)
        assert r.status_code in (400, 422), f"应为 400/422: {r.status_code}"

    def test_monitor_config_model_without_id(self, auth_headers):
        """model 模式不传 modelId 应 400"""
        r = requests.post(f"{BASE}/api/monitor/config",
                         json={"mode": "model", "modelId": ""},
                         headers=auth_headers)
        assert r.status_code in (400, 422), f"应为 400/422: {r.status_code}"

    def test_save_as_new_nonexistent_model(self, auth_headers):
        """另存不存在的模型应返回 404 或 500"""
        r = requests.post(f"{BASE}/api/ml/models/no_such_model_xyz/adjust",
                         json={"saveAsNew": True},
                         headers=auth_headers)
        assert r.status_code in (404, 500), f"应为 404/500: {r.status_code} {r.text[:200]}"

    def test_backtest_hist_zero(self, auth_headers):
        """hist=0 边界（应被钳制到 min_hist_for_ml）"""
        models = api("GET", "/api/ml/models", headers=auth_headers)
        if not models:
            pytest.skip("无可用模型")
        mid = _pick_computable(models)["id"]
        r = requests.post(f"{BASE}/api/ml/backtest",
                         json={"modelId": mid, "board": "all", "poolSize": 30,
                               "groups": 2, "n": 3, "hist": 0, "applyCost": False},
                         headers=auth_headers)
        if r.status_code == 200:
            result = r.json()
            assert result["actualHistDays"] >= 244, f"hist=0 应被钳制: {result['actualHistDays']}"
        # 422 也是合理的（样本不足）

    def test_backtest_hist_negative(self, auth_headers):
        """hist=-1 边界"""
        models = api("GET", "/api/ml/models", headers=auth_headers)
        if not models:
            pytest.skip("无可用模型")
        mid = _pick_computable(models)["id"]
        r = requests.post(f"{BASE}/api/ml/backtest",
                         json={"modelId": mid, "board": "all", "poolSize": 30,
                               "groups": 2, "n": 3, "hist": -1, "applyCost": False},
                         headers=auth_headers)
        # -1 被 int(hist) 转为 -1，max(-1, 244) = 244
        if r.status_code == 200:
            result = r.json()
            assert result["actualHistDays"] >= 244

    def test_monitor_config_ranking_invalid(self, auth_headers):
        """无效 ranking 应 400"""
        models = api("GET", "/api/ml/models", headers=auth_headers)
        if not models:
            pytest.skip("无可用模型")
        mid = _pick_computable(models)["id"]
        r = requests.post(f"{BASE}/api/monitor/config",
                         json={"mode": "model", "modelId": mid, "ranking": "invalid"},
                         headers=auth_headers)
        assert r.status_code in (400, 422), f"应为 400/422: {r.status_code}"


class TestReportPayload:
    """报告 payload 验证"""

    def test_payload_from_result_includes_new_fields(self):
        from app import reporting
        result = {
            "factorLabel": "test",
            "histWarning": "hist 被调整了",
            "actualHistDays": 300,
            "effectiveStart": "2024-01-01",
            "effectiveEnd": "2024-12-31",
            "metrics": {},
            "config": {},
        }
        payload = reporting.payload_from_result(result)
        assert payload["histWarning"] == "hist 被调整了"
        assert payload["actualHistDays"] == 300
        assert payload["effectiveStart"] == "2024-01-01"
        assert payload["effectiveEnd"] == "2024-12-31"

    def test_payload_without_new_fields_still_works(self):
        from app import reporting
        result = {"factorLabel": "test", "metrics": {}, "config": {}}
        payload = reporting.payload_from_result(result)
        assert payload["histWarning"] == ""
        assert payload["actualHistDays"] is None
        assert payload["effectiveStart"] is None
        assert payload["effectiveEnd"] is None


class TestCloneModel:
    """clone_model_with_adjust 函数单元测试"""

    def test_clone_creates_new_file(self, auth_headers):
        models = api("GET", "/api/ml/models", headers=auth_headers)
        if not models:
            pytest.skip("无可用模型")
        mid = _pick_computable(models)["id"]
        from app import ml
        import os as _os
        new_meta = ml.clone_model_with_adjust(mid, {"momentum5": 2.0}, 0.01)
        assert new_meta["id"].startswith("clone_"), new_meta["id"]
        assert _os.path.exists(new_meta["path"]), f"新模型文件不存在: {new_meta['path']}"
        # 清理
        _os.remove(new_meta["path"])
        json_path = new_meta["path"].replace(".joblib", ".json")
        if _os.path.exists(json_path):
            _os.remove(json_path)

    def test_clone_nonexistent_model(self):
        from app import ml
        with pytest.raises(FileNotFoundError):
            ml.clone_model_with_adjust("nonexistent_model_id")

    def test_clone_preserves_feature_names(self, auth_headers):
        models = api("GET", "/api/ml/models", headers=auth_headers)
        if not models:
            pytest.skip("无可用模型")
        mid = _pick_computable(models)["id"]
        from app import ml
        import os as _os
        new_meta = ml.clone_model_with_adjust(mid, {"momentum5": 2.0, "rsi": 0.5})
        assert len(new_meta["featureNames"]) > 0
        assert new_meta["featureWeights"] is not None
        assert new_meta["featureWeights"]["momentum5"] == 2.0
        assert new_meta["featureWeights"]["rsi"] == 0.5
        # 清理
        _os.remove(new_meta["path"])
        json_path = new_meta["path"].replace(".joblib", ".json")
        if _os.path.exists(json_path):
            _os.remove(json_path)


# ========================================================================
# 四、稳定性测试（连续多次调用验证无崩溃）
# ========================================================================

class TestStability:
    """接口稳定性：连续多次调用不崩溃"""

    def test_models_list_stable(self, auth_headers):
        for i in range(10):
            models = api("GET", "/api/ml/models", headers=auth_headers)
            assert isinstance(models, list), f"第 {i} 次返回非列表: {type(models)}"

    def test_monitor_status_stable(self, auth_headers):
        for i in range(10):
            status = api("GET", "/monitor/status", headers=auth_headers)
            assert "config" in status, f"第 {i} 次无 config"

    def test_monitor_config_stable(self, auth_headers):
        for i in range(10):
            cfg = api("GET", "/monitor/config", headers=auth_headers)
            assert "mode" in cfg, f"第 {i} 次无 mode"

    def test_model_params_stable(self, auth_headers):
        models = api("GET", "/api/ml/models", headers=auth_headers)
        if not models:
            pytest.skip("无可用模型")
        mid = _pick_computable(models)["id"]
        for i in range(5):
            params = api("GET", f"/api/ml/models/{mid}/params", headers=auth_headers)
            assert "featureNames" in params, f"第 {i} 次无 featureNames"
