"""中间结果落盘模块：选股/ML打分/回测/组合优化等模块产出 → 本地 JSON 文件。

联通设计：A 模块的产出保存为 artifact，B 模块可读取复用（如 ML 打分结果 →
选股 → 组合优化 → 风险归因），中间结果不依赖内存/DB，重启不丢。
"""
import os
import json
import uuid
import datetime

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def _path(aid: str) -> str:
    if "/" in aid or "\\" in aid or aid.startswith("."):
        raise ValueError("非法 artifact id")
    return os.path.join(ARTIFACT_DIR, f"{aid}.json")


def save_artifact(kind: str, payload: dict, name: str = "") -> dict:
    """保存中间结果到本地文件，返回元数据记录（不含 payload）。"""
    aid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    record = {
        "id": aid, "kind": kind, "name": name,
        "createdAt": datetime.datetime.now().isoformat(),
        "payload": payload,
    }
    with open(_path(aid), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, default=str)
    return {k: v for k, v in record.items() if k != "payload"}


def list_artifacts(kind: str | None = None, limit: int = 100) -> list[dict]:
    """按 kind 过滤列出最近 limit 条中间结果（不含 payload）。"""
    out = []
    if not os.path.isdir(ARTIFACT_DIR):
        return out
    for fn in sorted(os.listdir(ARTIFACT_DIR), reverse=True):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(ARTIFACT_DIR, fn), "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue
        if kind and rec.get("kind") != kind:
            continue
        out.append({k: v for k, v in rec.items() if k != "payload"})
        if len(out) >= limit:
            break
    return out


def load_artifact(aid: str) -> dict | None:
    try:
        p = _path(aid)
    except ValueError:
        return None
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def delete_artifact(aid: str) -> bool:
    try:
        p = _path(aid)
    except ValueError:
        return False
    if os.path.exists(p):
        os.remove(p)
        return True
    return False
