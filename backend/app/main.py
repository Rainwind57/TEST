"""FastAPI 应用入口。"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import db
from .routers import (quote, factor, portfolio, selection, strategies, reports,
                      jobs, ml, monitor, optimize, portfolio_opt, data, risk,
                      intraday, auth)

app = FastAPI(title="简易量化研究平台 API", version="3.3.0")

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


@app.on_event("startup")
def on_startup():
    db.init_db()


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
app.include_router(auth.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
