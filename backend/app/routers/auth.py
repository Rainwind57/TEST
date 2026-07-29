"""鉴权路由：注册、登录、当前用户。"""
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel

from .. import auth

router = APIRouter(prefix="/api/auth", tags=["auth"])


class Credentials(BaseModel):
    username: str
    password: str


@router.post("/register")
def register(body: Credentials):
    try:
        return auth.register_user(body.username, body.password)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/login")
def login(body: Credentials):
    try:
        return auth.login_user(body.username, body.password)
    except ValueError as e:
        raise HTTPException(401, str(e))


@router.get("/me")
def me(request: Request):
    uid = auth.get_user_id_from_auth(request.headers.get("Authorization"))
    if not uid:
        raise HTTPException(401, "未登录或 token 失效")
    import sqlite3
    from .. import db
    conn = db.get_conn()
    row = conn.execute("SELECT id, username, created_at FROM users WHERE id = ?", (uid,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "用户不存在")
    return dict(row)


def current_user_id(request: Request) -> int:
    """依赖注入：从请求头提取 user_id，未登录返回 0（匿名，兼容单机模式）。"""
    return auth.get_user_id_from_auth(request.headers.get("Authorization")) or 0
