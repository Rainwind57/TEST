"""盯盘看板路由：调度器开关、状态、信号、净值。"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from .. import scheduler, db
from .auth import require_user_id

router = APIRouter(prefix="/api/monitor", tags=["monitor"])


class ToggleBody(BaseModel):
    enabled: bool


class SignalConfigBody(BaseModel):
    mode: str  # rule=内置动量/RSI 规则 | model=落盘 ML 模型打分
    modelId: str = ""
    ranking: str = "isolated"      # 模型模式排名口径：isolated=孤立打分 / full=全池排名分位
    ruleFactor: str = ""          # 规则模式因子名（空=默认动量+RSI）
    board: str = "all"            # 模型模式候选板块（与选股口径对齐）
    poolSize: int = 150           # 模型模式候选池规模
    adjustId: str = ""            # 模型模式调参配置 artifact id


@router.get("/status")
def status():
    return {
        "enabled": scheduler.is_enabled(),
        "lastRun": scheduler.last_run(),
        "signals": scheduler.last_signals(),
        "config": scheduler.get_signal_config(),
    }


@router.get("/config")
def get_signal_config():
    return scheduler.get_signal_config()


@router.post("/config")
def set_signal_config(body: SignalConfigBody, uid: int = Depends(require_user_id)):
    """设置盯盘信号引擎：rule 规则或指定 ML 模型（modelId 必填）。
    
    模型模式：ranking=isolated 对各股孤立打分，ranking=full 对全池排名后取分位（与选股口径一致）。
    board/poolSize/adjustId 为模型模式下的候选板块/池规模/调参配置，与选股口径对齐。
    规则模式：ruleFactor 为空则默认动量+RSI，否则用指定因子。
    """
    try:
        return scheduler.set_signal_config(
            body.mode, body.modelId, body.ranking, body.ruleFactor,
            board=body.board, pool_size=body.poolSize, adjust_id=body.adjustId,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/toggle")
async def toggle(body: ToggleBody, uid: int = Depends(require_user_id)):
    # async 必须：AsyncIOScheduler.start() 需运行中事件循环，
    # 旧版同步路由在线程池跑 → get_event_loop() 抛错 → 前端 network error
    if body.enabled:
        scheduler.start()
    else:
        scheduler.stop()
    return {"enabled": scheduler.is_enabled()}


@router.post("/scan")
async def scan(force: bool = False, uid: int = Depends(require_user_id)):
    """立即手动扫描一次盯盘信号（无需等待交易日 15:10 的 cron）。

    force=true 时跳过交易日判断，任意时刻均可验证扫描逻辑。
    """
    return await scheduler.scan_now(force=force)


@router.get("/equity")
def equity_history(limit: int = 60):
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT ts, value FROM equity_history ORDER BY id DESC LIMIT ?",
        (max(1, min(limit, 500)),)
    ).fetchall()[::-1]
    conn.close()
    return [{"ts": r["ts"], "value": r["value"]} for r in rows]


class AutoTradeBody(BaseModel):
    enabled: bool


@router.post("/auto-trade")
def set_auto_trade(body: AutoTradeBody, uid: int = Depends(require_user_id)):
    db.set_setting("auto_trade", "1" if body.enabled else "0")
    return {"autoTrade": body.enabled}


@router.get("/auto-trade")
def get_auto_trade():
    return {"autoTrade": db.get_setting("auto_trade", "0") == "1"}
