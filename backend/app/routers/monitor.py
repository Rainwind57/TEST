"""盯盘看板路由：调度器开关、状态、信号、净值。"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import scheduler, db

router = APIRouter(prefix="/api/monitor", tags=["monitor"])


class ToggleBody(BaseModel):
    enabled: bool


@router.get("/status")
def status():
    return {
        "enabled": scheduler.is_enabled(),
        "lastRun": scheduler.last_run(),
        "signals": scheduler.last_signals(),
    }


@router.post("/toggle")
async def toggle(body: ToggleBody):
    # async 必须：AsyncIOScheduler.start() 需运行中事件循环，
    # 旧版同步路由在线程池跑 → get_event_loop() 抛错 → 前端 network error
    if body.enabled:
        scheduler.start()
    else:
        scheduler.stop()
    return {"enabled": scheduler.is_enabled()}


@router.get("/equity")
def equity_history(limit: int = 60):
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT ts, value FROM equity_history ORDER BY id DESC LIMIT ?",
        (max(1, min(limit, 500)),)
    ).fetchall()[::-1]
    conn.close()
    return [{"ts": r["ts"], "value": r["value"]} for r in rows]
