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
from collections import defaultdict, deque

from . import db

# 登录撞库防护：双层限流 —— 按 IP + 按账号
_LOGIN_ATTEMPTS_IP: dict[str, deque] = defaultdict(deque)
_LOGIN_ATTEMPTS_USER: dict[str, deque] = defaultdict(deque)
_LOGIN_MAX_ATTEMPTS = 5
_LOGIN_WINDOW = 60.0

# 注册防刷：每 IP 每小时最多 3 次注册
_REGISTER_ATTEMPTS: dict[str, deque] = defaultdict(deque)
_REGISTER_MAX = 3
_REGISTER_WINDOW = 3600.0


def _check_login_rate(client_ip: str, username: str) -> str | None:
    """检查登录频率，返回 None 表示放行，否则返回错误消息。按 IP + 按账号双层校验。"""
    now = time.time()
    for label, dq, limit in [
        (f"IP {client_ip}", _LOGIN_ATTEMPTS_IP[client_ip], _LOGIN_MAX_ATTEMPTS),
        (f"账号 {username}", _LOGIN_ATTEMPTS_USER[username], _LOGIN_MAX_ATTEMPTS),
    ]:
        while dq and now - dq[0] > _LOGIN_WINDOW:
            dq.popleft()
        if len(dq) >= limit:
            return f"{label} 登录请求过于频繁，请稍后重试"
    _LOGIN_ATTEMPTS_IP[client_ip].append(now)
    _LOGIN_ATTEMPTS_USER[username].append(now)
    return None


def _clear_login_rate(client_ip: str, username: str) -> None:
    """登录成功后清除该用户的撞库计数器。"""
    _LOGIN_ATTEMPTS_IP.pop(client_ip, None)
    _LOGIN_ATTEMPTS_USER.pop(username, None)


def _check_register_rate(client_ip: str) -> bool:
    """检查注册频率，超限返回 False。"""
    now = time.time()
    dq = _REGISTER_ATTEMPTS[client_ip]
    while dq and now - dq[0] > _REGISTER_WINDOW:
        dq.popleft()
    if len(dq) >= _REGISTER_MAX:
        return False
    dq.append(now)
    return True

# JWT 密钥：优先读环境变量 QUANT_JWT_SECRET；未设时生成临时随机密钥（不落盘）。
# 旧版把密钥明文持久化进 SQLite settings 表，拿到 quant.db 即可伪造任意用户 token；
# 改为不落盘后，未设环境变量时重启会使已签发 token 失效（安全优先，生产务必注入环境变量）。
JWT_SECRET = os.environ.get("QUANT_JWT_SECRET") or secrets.token_hex(32)


def ensure_secret_persisted() -> None:
    """兼容旧调用点：密钥不再持久化，此函数保留为空实现。"""
    return
JWT_TTL = 7 * 24 * 3600  # 7 天


_PBKDF2_ITERATIONS = 600_000  # OWASP 2023 建议 ≥60 万次
_PBKDF2_ITERATIONS_LEGACY = 100_000  # 存量用户旧迭代次数，登录成功后自动升级


def _hash_password(password: str, salt: str, iterations: int = _PBKDF2_ITERATIONS) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), iterations).hex()


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


def register_user(username: str, password: str, client_ip: str = "") -> dict:
    if client_ip and not _check_register_rate(client_ip):
        raise ValueError("注册过于频繁，请稍后再试")
    if not username or len(username) < 3:
        raise ValueError("用户名至少 3 个字符")
    if len(password) < 6:
        raise ValueError("密码至少 6 个字符")
    with db.get_conn() as conn:
        existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
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
    return {"id": uid, "username": username, "token": _make_token(uid, username)}


def login_user(username: str, password: str, client_ip: str = "") -> dict:
    if client_ip:
        err = _check_login_rate(client_ip, username)
        if err:
            raise ValueError(err)
    with db.get_conn() as conn:
        row = conn.execute("SELECT id, username, password, salt FROM users WHERE username = ?",
                          (username,)).fetchone()
    if not row:
        raise ValueError("用户不存在或密码错误")
    pwd_hash = _hash_password(password, row["salt"])
    if not hmac.compare_digest(pwd_hash, row["password"]):
        # 兼容旧迭代次数（10 万）的存量密码：验证通过后升级到新迭代
        legacy_hash = _hash_password(password, row["salt"], _PBKDF2_ITERATIONS_LEGACY)
        if not hmac.compare_digest(legacy_hash, row["password"]):
            raise ValueError("用户不存在或密码错误")
        try:
            with db.get_conn() as conn2:
                conn2.execute("UPDATE users SET password = ? WHERE id = ?", (pwd_hash, row["id"]))
                conn2.commit()
        except Exception:
            pass
    if client_ip:
        _clear_login_rate(client_ip, username)
    return {"id": row["id"], "username": row["username"], "token": _make_token(row["id"], row["username"])}


def get_user_id_from_auth(auth_header: str | None) -> int | None:
    """从 Authorization: Bearer <token> 提取 user_id，失败返回 None。"""
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header[7:]
    payload = verify_token(token)
    return payload.get("uid") if payload else None


def get_user_id_from_request(request) -> int | None:
    """从请求提取 user_id：优先 Authorization 头，其次 quant_token httpOnly Cookie。

    Cookie 为持久化凭证（前端不再把 token 写入 localStorage，降低 XSS 窃取面）。
    """
    uid = get_user_id_from_auth(request.headers.get("Authorization"))
    if uid:
        return uid
    cookie_token = request.cookies.get("quant_token")
    if cookie_token:
        payload = verify_token(cookie_token)
        if payload:
            return payload.get("uid")
    return None
