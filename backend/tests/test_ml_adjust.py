"""ml.py 人工调参工具函数单元测试。

覆盖：
- _apply_feature_weights 特征权重缩放
- _apply_threshold 阈值偏移（单调，不改变排序）
"""
import numpy as np
import pytest

from app import ml


def test_apply_feature_weights_scales_column():
    """特征权重 2.0 → 该特征列数值翻倍，其余列不变。"""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    names = ["a", "b"]
    out = ml._apply_feature_weights(X, names, {"a": 2.0})
    assert out.shape == X.shape
    assert out[0].tolist() == [2.0, 2.0]
    assert out[1].tolist() == [6.0, 4.0]
    assert out[2].tolist() == [10.0, 6.0]


def test_apply_feature_weights_unknown_feature_ignored():
    """不存在的特征名应被忽略（不崩、不改数据）。"""
    X = np.array([[1.0, 2.0]])
    out = ml._apply_feature_weights(X, ["a", "b"], {"nonexistent": 5.0})
    assert out[0].tolist() == [1.0, 2.0]


def test_apply_feature_weights_empty_returns_copy():
    """无权重配置时返回原值拷贝，不改动输入。"""
    X = np.array([[1.0, 2.0]])
    out = ml._apply_feature_weights(X, ["a", "b"], {})
    assert out[0].tolist() == [1.0, 2.0]


def test_apply_threshold_shifts_values():
    """阈值 +0.01 → 每个预测分 +0.01。"""
    preds = [0.5, -0.2, 0.0]
    out = ml._apply_threshold(preds, 0.01)
    assert out == pytest.approx([0.51, -0.19, 0.01])


def test_apply_threshold_none_unchanged():
    assert ml._apply_threshold([0.5, -0.2], None) == [0.5, -0.2]


def test_apply_threshold_preserves_order():
    """阈值是单调变换，排序应与原一致（分层回测不受影响）。"""
    preds = [0.3, 0.8, -0.1, 0.5]
    out = ml._apply_threshold(preds, 0.05)
    assert np.argsort(out).tolist() == np.argsort(preds).tolist()


# ---------------- 路由层 _load_adjust ----------------

def test_load_adjust_from_dict():
    """直传 adjust dict 应原样返回。"""
    from app.routers import ml as ml_router
    cfg = ml_router._load_adjust(None, {"modelId": "m1", "featureWeights": {"a": 2.0}})
    assert cfg["featureWeights"] == {"a": 2.0}


def test_load_adjust_from_artifact(tmp_path, monkeypatch):
    """adjustId 应从 artifact 读取配置。"""
    from app.routers import ml as ml_router
    from app import artifacts
    monkeypatch.setattr(artifacts, "ARTIFACT_DIR", str(tmp_path))
    meta = artifacts.save_artifact("ml_adjust", {
        "modelId": "m1", "featureNames": ["a", "b"],
        "featureWeights": {"a": 2.0}, "threshold": 0.01,
    })
    cfg = ml_router._load_adjust(meta["id"], None)
    assert cfg["featureWeights"] == {"a": 2.0}
    assert cfg["threshold"] == 0.01


def test_load_adjust_missing_artifact(tmp_path, monkeypatch):
    """不存在的 adjustId 应 404。"""
    import fastapi
    from app.routers import ml as ml_router
    from app import artifacts
    monkeypatch.setattr(artifacts, "ARTIFACT_DIR", str(tmp_path))
    with pytest.raises(fastapi.HTTPException) as exc:
        ml_router._load_adjust("no_such_id", None)
    assert exc.value.status_code == 404


def test_load_adjust_none():
    """均未传时应返回 None（模型原样推理）。"""
    from app.routers import ml as ml_router
    assert ml_router._load_adjust(None, None) is None
