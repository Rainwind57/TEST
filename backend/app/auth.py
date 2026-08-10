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

# JWT 密钥：优先读环境变量 QUANT_JWT_SECRET；未设时从 DB settings 表读取持久化密钥；
# 都不存在时随机生成并持久化，避免重启后 token 全部失效（P2修复）。
_JWT_ENV = os.environ.get("QUANT_JWT_SECRET")
if _JWT_ENV:
    JWT_SECRET = _JWT_ENV
else:
    try:
        _saved = db.get_setting("jwt_secret", "")
        if _saved:
            JWT_SECRET = _saved
        else:
            JWT_SECRET = secrets.token_hex(32)
            db.set_setting("jwt_secret", JWT_SECRET)
    except Exception:
        JWT_SECRET = secrets.token_hex(32)
        warnings.warn(
            "QUANT_JWT_SECRET 未设置且 DB 不可用，使用临时随机密钥（启动后 init_db 完成会自动持久化）",
            RuntimeWarning, stacklevel=2,
        )


def ensure_secret_persisted() -> None:
    """DB 初始化完成后调用：若密钥为临时生成且未持久化，则写入 settings 表。"""
    if _JWT_ENV:
        return
    try:
        if db.get_setting("jwt_secret", "") == "":
            db.set_setting("jwt_secret", JWT_SECRET)
    except Exception:
        pass
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


def register_user(username: str, password: str, client_ip: str = "") -> dict:
    if client_ip and not _check_register_rate(client_ip):
        raise ValueError("注册过于频繁，请稍后再试")
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


def login_user(username: str, password: str, client_ip: str = "") -> dict:
    if client_ip:
        err = _check_login_rate(client_ip, username)
        if err:
            raise ValueError(err)
    conn = db.get_conn()
    row = conn.execute("SELECT id, username, password, salt FROM users WHERE username = ?",
                      (username,)).fetchone()
    conn.close()
    if not row:
        raise ValueError("用户不存在或密码错误")
    if not hmac.compare_digest(_hash_password(password, row["salt"]), row["password"]):
        raise ValueError("用户不存在或密码错误")
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
