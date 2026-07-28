"""行情与自选股路由。"""
import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import adapters, db

router = APIRouter(prefix="/api", tags=["quote"])


class WatchAddBody(BaseModel):
    code: str


@router.get("/quote")
async def get_quote(codes: str, source: str = "tencent"):
    code_list = [c.strip() for c in codes.split(",") if c.strip()]
    if not code_list:
        raise HTTPException(400, "codes 不能为空")
    try:
        data = await adapters.fetch_quotes(code_list, source)
    except Exception as e:
        raise HTTPException(502, f"数据源请求失败: {e}")
    return data


@router.get("/kline")
async def get_kline(code: str, days: int = 150):
    try:
        data = await adapters.fetch_kline(code, days)
    except Exception as e:
        raise HTTPException(502, f"历史数据请求失败: {e}")
    return {"code": code, "data": data}


@router.get("/watchlist")
def get_watchlist():
    conn = db.get_conn()
    rows = conn.execute("SELECT code FROM watchlist ORDER BY added_at").fetchall()
    conn.close()
    return [r["code"] for r in rows]


@router.post("/watchlist")
def add_watchlist(body: WatchAddBody):
    code = body.code.strip().lower()
    conn = db.get_conn()
    existing = conn.execute("SELECT 1 FROM watchlist WHERE code = ?", (code,)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(409, "已在自选列表中")
    conn.execute("INSERT INTO watchlist (code, added_at) VALUES (?, ?)",
                 (code, datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return {"ok": True}


@router.delete("/watchlist/{code}")
def delete_watchlist(code: str):
    conn = db.get_conn()
    conn.execute("DELETE FROM watchlist WHERE code = ?", (code,))
    conn.commit()
    conn.close()
    return {"ok": True}
