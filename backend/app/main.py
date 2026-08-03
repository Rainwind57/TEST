"""FastAPI 应用入口。"""
import os
import time
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from . import db, auth
from .logging_config import setup_logging, get_request_logger
from .routers import (quote, factor, portfolio, selection, strategies, reports,
                      jobs, ml, monitor, optimize, portfolio_opt, data, risk,
                      intraday, artifacts, auth as auth_router)

app = FastAPI(title="简易量化研究平台 API", version="3.3.0")

# 结构化日志（P2-11）：JSON 格式 + stdout，容器环境友好
setup_logging(level=os.environ.get("QUANT_LOG_LEVEL", "INFO"))
logger = get_request_logger()

# CORS 白名单环境变量化：旧版硬编码 localhost:5899，换 IP/端口访问被浏览器拦，
# 体感“连不上”。QUANT_CORS_ORIGINS 逗号分隔，未设时保留本地默认。
_DEFAULT_CORS = "http://localhost:5899,http://127.0.0.1:5899,http://localhost:5173,http://127.0.0.1:5173"
_cors_env = os.environ.get("QUANT_CORS_ORIGINS", _DEFAULT_CORS)
_cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 读接口限流：旧版 /api/quote、/api/select/backtest、/api/factor 等读接口完全开放，
# 同网段任意客户端可调用算力与第三方接口被刷。按 IP 维护滑动窗口，登录用户宽松、
# 匿名严格；写操作本就 require_user_id 强制登录，不限流。
_ANON_LIMIT = int(os.environ.get("QUANT_RATE_ANON", "60"))   # 匿名每分钟
_USER_LIMIT = int(os.environ.get("QUANT_RATE_USER", "600"))  # 登录每分钟
_WINDOW = 60.0
_rate_buckets: dict[str, deque] = defaultdict(deque)
_exempt_paths = {"/api/health", "/api/auth/login", "/api/auth/register"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in _exempt_paths or not path.startswith("/api/"):
            return await call_next(request)
        ip = request.client.host if request.client else "unknown"
        token = request.headers.get("Authorization", "")
        uid = auth.get_user_id_from_auth(token) if token.startswith("Bearer ") else None
        limit = _USER_LIMIT if uid else _ANON_LIMIT
        key = f"{ip}:{'u' if uid else 'a'}"
        now = time.time()
        dq = _rate_buckets[key]
        while dq and now - dq[0] > _WINDOW:
            dq.popleft()
        if len(dq) >= limit:
            return JSONResponse(
                status_code=429,
                content={"detail": f"请求过于频繁，每分钟上限 {limit} 次"},
            )
        dq.append(now)
        return await call_next(request)


app.add_middleware(RateLimitMiddleware)


# 请求日志中间件（P2-11）：记录每个请求的方法/路径/状态/耗时，便于审计与性能分析
import uuid
import time as _time


class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        start = _time.time()
        response = await call_next(request)
        elapsed_ms = int((_time.time() - start) * 1000)
        response.headers["X-Request-ID"] = request_id
        # 仅记 API 请求，跳过健康检查噪音
        path = request.url.path
        if path.startswith("/api/") and path != "/api/health":
            logger.info(
                f"{request.method} {path} {response.status_code} {elapsed_ms}ms",
                extra={"request_id": request_id}
            )
        return response


app.add_middleware(RequestLogMiddleware)


@app.on_event("startup")
async def on_startup():
    db.init_db()
    auth.ensure_secret_persisted()
    # P0修复：启动时自动拉起调度器（若上次持久化为 enabled）
    from . import scheduler
    if db.get_scheduler_enabled():
        scheduler.start()


app.include_router(quote.router)
app.include_router(factor.router)
app.include_router(portfolio.router)
app.include_router(selection.router)
app.include_router(strategies.router)
app.include_router(reports.router)
app.include_router(jobs.router)
app.include_router(ml.router)
app.include_router(monitor.router)
app.include_router(optimize.router)
app.include_router(portfolio_opt.router)
app.include_router(data.router)
app.include_router(risk.router)
app.include_router(intraday.router)
app.include_router(artifacts.router)
app.include_router(auth_router.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
