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
async def get_kline(code: str, days: int = 500, forceRefresh: bool = False):
    days = max(30, min(days, 2000))  # 防止 days=99999 导致的请求爆炸
    try:
        data = await adapters.fetch_kline(code, days, force_refresh=forceRefresh)
    except Exception as e:
        raise HTTPException(502, f"历史数据请求失败: {e}")
    return {"code": code, "data": data}


@router.get("/stock/exists")
async def stock_exists(code: str):
    """校验股票代码是否存在：依次查腾讯/新浪/东财，任一有。

    返回 {"exists": bool, "name": str}；所有数据源异常时 exists=true（无法确认则放行，
    避免网络抖动误拦真实股票）。
    """
    code = code.strip().lower()
    queried_ok = False
    for fn in (adapters.fetch_tencent_quotes, adapters.fetch_sina_quotes,
               adapters.fetch_eastmoney_quotes):
        try:
            data = await fn([code])
            queried_ok = True
            if data.get(code):
                return {"exists": True, "name": data[code].get("name", "")}
        except Exception:
            continue
    return {"exists": not queried_ok, "name": ""}


@router.get("/watchlist")
def get_watchlist():
    conn = db.get_conn()
    rows = conn.execute("SELECT code FROM watchlist ORDER BY added_at").fetchall()
    conn.close()
    return [r["code"] for r in rows]


@router.post("/watchlist")
def add_watchlist(body: WatchAddBody):
    code = body.code.strip().lower()
    if not db.is_tradable(code):
        raise HTTPException(400, f"无法添加非交易标的（{code} 为指数/ETF），请选择可交易个股")
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


@router.get("/timeshare")
async def get_timeshare(code: str):
    """当日分时图：1分钟价格+成交量+均价黄线。"""
    try:
        data = await adapters.fetch_time_share(code)
    except Exception as e:
        raise HTTPException(502, f"分时数据请求失败: {e}")
    return {"code": code, "data": data}
