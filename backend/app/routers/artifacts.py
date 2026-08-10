"""中间结果管理路由：选股/ML打分/回测/组合优化等产出可保存/读取/复用。

联通设计：前端完成"选股→回测→组合→风险"任一环节后，把产出保存为 artifact；
下一环节可从 /api/artifacts/{id} 读取上一环节的结果（如选股结果的 codes 直接
喂给风险归因），形成可复现的研究流水线。
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from .. import artifacts
from .auth import require_user_id

router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])


class SaveBody(BaseModel):
    kind: str
    payload: dict
    name: str = ""


@router.post("")
def save_artifact(body: SaveBody, uid: int = Depends(require_user_id)):
    if not body.payload:
        raise HTTPException(400, "payload 不能为空")
    if not body.kind.strip():
        raise HTTPException(400, "kind 不能为空")
    return artifacts.save_artifact(body.kind.strip(), body.payload, body.name.strip(), user_id=uid)


@router.get("")
def list_artifacts(kind: str | None = None, limit: int = 100, uid: int = Depends(require_user_id)):
    return artifacts.list_artifacts(kind, max(1, min(limit, 500)), user_id=uid)


@router.get("/{aid}")
def get_artifact(aid: str, uid: int = Depends(require_user_id)):
    rec = artifacts.load_artifact(aid, user_id=uid)
    if not rec:
        raise HTTPException(404, "中间结果不存在")
    return rec


@router.delete("/{aid}")
def remove_artifact(aid: str, uid: int = Depends(require_user_id)):
    if not artifacts.delete_artifact(aid, user_id=uid):
        raise HTTPException(404, "中间结果不存在")
    return {"ok": True}
