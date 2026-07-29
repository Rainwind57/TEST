"""FastAPI 应用入口。"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import db
from .routers import quote, factor, portfolio, selection, strategies, reports

app = FastAPI(title="简易量化研究平台 API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5899", "http://127.0.0.1:5899"],
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


@app.get("/api/health")
def health():
    return {"status": "ok"}
