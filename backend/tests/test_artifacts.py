"""artifacts.py 中间结果落盘单元测试。

覆盖：
- save/list/load/delete 完整生命周期
- kind 过滤
- 非法 id 防御（路径穿越）
"""
import pytest

from app import artifacts


@pytest.fixture
def art_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "ARTIFACT_DIR", str(tmp_path))
    return tmp_path


def test_save_and_load_roundtrip(art_dir):
    meta = artifacts.save_artifact("select", {"codes": ["sh600519"], "rows": []}, name="测试")
    assert meta["id"]
    assert meta["kind"] == "select"
    assert meta["name"] == "测试"
    assert "payload" not in meta  # 元数据不含 payload

    rec = artifacts.load_artifact(meta["id"])
    assert rec["payload"]["codes"] == ["sh600519"]
    assert rec["kind"] == "select"


def test_list_by_kind(art_dir):
    artifacts.save_artifact("select", {"codes": []}, name="a")
    artifacts.save_artifact("backtest", {"metrics": {}}, name="b")
    artifacts.save_artifact("select", {"codes": []}, name="c")

    all_arts = artifacts.list_artifacts()
    assert len(all_arts) == 3
    selects = artifacts.list_artifacts("select")
    assert len(selects) == 2
    assert all(a["kind"] == "select" for a in selects)


def test_list_limit(art_dir):
    for i in range(5):
        artifacts.save_artifact("test", {"i": i})
    assert len(artifacts.list_artifacts(limit=2)) == 2


def test_delete(art_dir):
    meta = artifacts.save_artifact("test", {"x": 1})
    assert artifacts.delete_artifact(meta["id"]) is True
    assert artifacts.load_artifact(meta["id"]) is None
    assert artifacts.delete_artifact(meta["id"]) is False  # 已删再删返回 False


def test_load_missing(art_dir):
    assert artifacts.load_artifact("nonexistent_id") is None


def test_invalid_id_path_traversal(art_dir):
    """路径穿越防御：../ 等非法 id 应拒绝而非读写任意路径。"""
    assert artifacts.load_artifact("../quant.db") is None
    assert artifacts.delete_artifact("../quant.db") is False
    with pytest.raises(ValueError):
        artifacts._path("../evil.json")
