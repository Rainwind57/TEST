"""用户鉴权（JWT）：注册/登录/token 校验。

轻量多用户：saved_strategies / user_factors / backtest_runs 按 user_id 隔离，
watchlist / portfolio 等行情工具保持单机共享。
密码用 hashlib + 盐做 PBKDF2 风格哈希（不引入 passlib 重依赖）。
"""
import os
import hmac
import hashlib
import secrets
import datetime
import time
import warnings

from . import db

# JWT 密钥：优先环境变量 QUANT_JWT_SECRET；未设时用固定默认值（保证重启后 token 仍有效，
# 旧版每次启动随机生成会导致所有已签发 token 失效）。生产部署必须设置环境变量。
_JWT_ENV = os.environ.get("QUANT_JWT_SECRET")
if _JWT_ENV:
    JWT_SECRET = _JWT_ENV
else:
    JWT_SECRET = "dev-only-insecure-secret-please-set-QUANT_JWT_SECRET"
    warnings.warn(
        "QUANT_JWT_SECRET 未设置，使用不安全的开发默认值；生产环境请设置该环境变量",
        RuntimeWarning,
        stacklevel=2,
    )
JWT_TTL = 7 * 24 * 3600  # 7 天


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000).hex()


def _make_token(user_id: int, username: str) -> str:
    """简易 JWT（HMAC-SHA256 签名，不依赖 PyJWT）。"""
    payload = {"uid": user_id, "sub": username, "exp": int(time.time()) + JWT_TTL}
    import json
    body = json.dumps(payload, separators=(",", ":"))
    body_b64 = _b64url(body.encode())
    sig = hmac.new(JWT_SECRET.encode(), body_b64.encode(), hashlib.sha256).digest()
    return f"{body_b64}.{_b64url(sig)}"


def verify_token(token: str) -> dict | None:
    """校验 token，返回 payload 或 None。"""
    if not token or "." not in token:
        return None
    import json
    body_b64, sig_b64 = token.split(".", 1)
    expected = hmac.new(JWT_SECRET.encode(), body_b64.encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(_b64url(expected), sig_b64):
        return None
    try:
        payload = json.loads(_b64url_decode(body_b64))
    except (ValueError, json.JSONDecodeError):
        return None
    if payload.get("exp", 0) < time.time():
        return None
    return payload


def _b64url(b: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    import base64
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def register_user(username: str, password: str) -> dict:
    if not username or len(username) < 3:
        raise ValueError("用户名至少 3 个字符")
    if len(password) < 6:
        raise ValueError("密码至少 6 个字符")
    conn = db.get_conn()
    existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if existing:
        conn.close()
        raise ValueError("用户名已存在")
    salt = secrets.token_hex(8)
    pwd_hash = _hash_password(password, salt)
    now = datetime.datetime.now().isoformat()
    cur = conn.execute(
        "INSERT INTO users (username, password, salt, created_at) VALUES (?, ?, ?, ?)",
        (username, pwd_hash, salt, now)
    )
    conn.commit()
    uid = cur.lastrowid
    conn.close()
    return {"id": uid, "username": username, "token": _make_token(uid, username)}


def login_user(username: str, password: str) -> dict:
    conn = db.get_conn()
    row = conn.execute("SELECT id, username, password, salt FROM users WHERE username = ?",
                      (username,)).fetchone()
    conn.close()
    if not row:
        raise ValueError("用户不存在或密码错误")
    if _hash_password(password, row["salt"]) != row["password"]:
        raise ValueError("用户不存在或密码错误")
    return {"id": row["id"], "username": row["username"], "token": _make_token(row["id"], row["username"])}


def get_user_id_from_auth(auth_header: str | None) -> int | None:
    """从 Authorization: Bearer <token> 提取 user_id，失败返回 None。"""
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header[7:]
    payload = verify_token(token)
    return payload.get("uid") if payload else None
