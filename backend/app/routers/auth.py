"""鉴权路由：注册、登录、当前用户。"""
from fastapi import APIRouter, HTTPException, Depends, Request, Response
from pydantic import BaseModel

from .. import auth

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _client_ip(request: Request) -> str:
    """取客户端真实 IP：经反代/Nginx 时用 X-Forwarded-For，否则回退 socket 直连地址。"""
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else ""


def _set_auth_cookie(response: Response, token: str) -> None:
    """httpOnly + SameSite=Lax：JS 不可读，降低 XSS 窃取持久化凭证的风险。"""
    response.set_cookie("quant_token", token, httponly=True, samesite="lax",
                        max_age=auth.JWT_TTL, path="/")


class Credentials(BaseModel):
    username: str
    password: str


@router.post("/register")
def register(body: Credentials, request: Request, response: Response):
    try:
        res = auth.register_user(body.username, body.password, client_ip=_client_ip(request))
        _set_auth_cookie(response, res["token"])
        return res
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/login")
def login(body: Credentials, request: Request, response: Response):
    try:
        res = auth.login_user(body.username, body.password, client_ip=_client_ip(request))
        _set_auth_cookie(response, res["token"])
        return res
    except ValueError as e:
        raise HTTPException(401, str(e))


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie("quant_token", path="/")
    return {"ok": True}


@router.get("/me")
def me(request: Request):
    uid = auth.get_user_id_from_request(request)
    if not uid:
        raise HTTPException(401, "未登录或 token 失效")
    import sqlite3
    from .. import db
    with db.get_conn() as conn:
        row = conn.execute("SELECT id, username, created_at FROM users WHERE id = ?", (uid,)).fetchone()
    if not row:
        raise HTTPException(404, "用户不存在")
    return dict(row)


def require_user_id(request: Request) -> int:
    """依赖注入：强制登录，未登录抛 401。用于敏感写操作（下单/删模型/策略保存等），
    防止 API 层被直接 curl 调用（旧版除 reset/me 外全部开放）。"""
    uid = auth.get_user_id_from_request(request)
    if not uid:
        raise HTTPException(401, "该操作需要登录")
    return uid
