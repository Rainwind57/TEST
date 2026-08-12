"""行情与自选股路由。"""
import datetime
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from .. import adapters, db
from .auth import require_user_id

router = APIRouter(prefix="/api", tags=["quote"])


class WatchAddBody(BaseModel):
    code: str
    name: str = ""


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
async def get_kline(code: str, days: int = 500, forceRefresh: bool = False, freq: str = "D"):
    days = max(30, min(days, 2000))  # 防止 days=99999 导致的请求爆炸
    try:
        data = await adapters.fetch_kline(code, days, force_refresh=forceRefresh)
    except Exception as e:
        raise HTTPException(502, f"历史数据请求失败: {e}")
    if freq in ("W", "M"):
        data = _convert_kline_freq(data, freq)
    return {"code": code, "data": data}


def _convert_kline_freq(daily: list[dict], freq: str) -> list[dict]:
    """日K聚合成周K/月K。"""
    from datetime import datetime
    if not daily:
        return []
    grouped: dict[str, dict] = {}
    for row in daily:
        d = datetime.strptime(row["date"][:10], "%Y-%m-%d")
        grp = d.strftime("%G-W%V") if freq == "W" else d.strftime("%Y-%m")
        o = float(row["open"])
        c = float(row["close"])
        h = float(row["high"])
        l = float(row["low"])
        v = float(row["volume"])
        if grp not in grouped:
            grouped[grp] = {"open": o, "close": c, "high": h, "low": l, "volume": v, "date": row["date"]}
        else:
            g = grouped[grp]
            g["close"] = c
            g["high"] = max(g["high"], h)
            g["low"] = min(g["low"], l)
            g["volume"] += v
            g["date"] = row["date"]
    return [grouped[k] for k in sorted(grouped.keys())]


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
    rows = conn.execute("SELECT code, name FROM watchlist ORDER BY added_at").fetchall()
    conn.close()
    return [{"code": r["code"], "name": r["name"] or ""} for r in rows]


@router.post("/watchlist")
def add_watchlist(body: WatchAddBody, uid: int = Depends(require_user_id)):
    code = body.code.strip().lower()
    if not db.is_tradable(code):
        raise HTTPException(400, f"无法添加非交易标的（{code} 为指数/ETF），请选择可交易个股")
    conn = db.get_conn()
    existing = conn.execute("SELECT 1 FROM watchlist WHERE code = ?", (code,)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(409, "已在自选列表中")
    conn.execute("INSERT INTO watchlist (code, name, added_at) VALUES (?, ?, ?)",
                 (code, (body.name or "").strip(), datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return {"ok": True}


@router.delete("/watchlist/{code}")
def delete_watchlist(code: str, uid: int = Depends(require_user_id)):
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
    if not data:
        import datetime
        now = datetime.datetime.now()
        is_trading = now.weekday() < 5 and datetime.time(9, 15) <= now.time() <= datetime.time(15, 0)
        hint = "当日分时数据为空，可能原因：非交易日、盘前盘后时段、或数据源暂不可用" if not is_trading else "当日暂无分时数据，请确认股票代码有效且处于交易时段"
        return {"code": code, "data": data, "warning": hint}
    return {"code": code, "data": data}
