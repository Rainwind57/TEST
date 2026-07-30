"""定时盯盘调度器：APScheduler 后台任务。

功能：
- 交易日 15:05 自动刷新持仓市值并写 equity_history（日频净值完整）
- 交易日 15:10 重算 watchlist 因子打分，生成信号（可选自动调仓，默认关闭）

默认关闭，需用户显式开启（模拟边界）。内存状态，重启即停。
"""
import asyncio
import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from . import db, adapters
from .routers import portfolio as pf
from .factors import FACTORS
from .numpy_factors import kline_to_arrays, compute_factor_series, series_at

_scheduler: AsyncIOScheduler | None = None
_enabled = False
_last_run = None
_last_signals = []


def is_enabled() -> bool:
    return _enabled


def last_run():
    return _last_run


def last_signals():
    return _last_signals


def start():
    global _scheduler, _enabled
    if _scheduler:
        return
    _scheduler = AsyncIOScheduler()
    _scheduler.add_job(_refresh_equity, CronTrigger(hour=15, minute=5, day_of_week="mon-fri"))
    _scheduler.add_job(_scan_signals, CronTrigger(hour=15, minute=10, day_of_week="mon-fri"))
    _scheduler.start()
    _enabled = True


def stop():
    global _scheduler, _enabled
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
    _enabled = False


# A 股休市节假日（需每年更新；接入正式交易日历接口后可替换此硬编码）
_HOLIDAYS = {
    "2026-01-01",
    "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20",
    "2026-04-05", "2026-04-06", "2026-04-07",
    "2026-05-01", "2026-05-02", "2026-05-03",
    "2026-06-19", "2026-06-20", "2026-06-21",
    "2026-09-25", "2026-09-26", "2026-09-27",
    "2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07",
}


def _is_trading_day() -> bool:
    """交易日判断：周末 + 节假日休市（节假日列表需年度更新，近似）。"""
    now = datetime.datetime.now()
    if now.weekday() >= 5:
        return False
    return now.strftime("%Y-%m-%d") not in _HOLIDAYS


async def _refresh_equity():
    if not _is_trading_day():
        return
    global _last_run
    try:
        conn = db.get_conn()
        state = conn.execute("SELECT cash FROM portfolio_state WHERE id = 1").fetchone()
        positions = conn.execute("SELECT code, name, qty, avg_cost FROM positions").fetchall()
        conn.close()
        if not state:
            return
        cash = state["cash"]
        codes = [r["code"] for r in positions] or ["sh000001"]
        quotes = await adapters.fetch_quotes(codes)
        market_value = 0.0
        for p in positions:
            q = quotes.get(p["code"])
            price = q.get("price", 0) if q else 0
            market_value += price * p["qty"]
        total = cash + market_value
        conn = db.get_conn()
        conn.execute("INSERT INTO equity_history (ts, value) VALUES (?, ?)",
                     (datetime.datetime.now().isoformat(), total))
        conn.commit()
        conn.close()
        _last_run = {"task": "refresh_equity", "ts": datetime.datetime.now().isoformat(),
                     "totalValue": total, "cash": cash, "marketValue": market_value}
    except Exception as e:
        _last_run = {"task": "refresh_equity", "error": str(e)}


async def _scan_signals():
    if not _is_trading_day():
        return
    global _last_signals
    try:
        conn = db.get_conn()
        codes = [r["code"] for r in conn.execute("SELECT code FROM watchlist ORDER BY added_at").fetchall()]
        conn.close()
        if not codes:
            _last_signals = []
            return
        signals = []
        for code in codes[:20]:
            try:
                kline = await adapters.fetch_kline(code, 60)
            except Exception:
                continue
            if len(kline) < 25:
                continue
            arr = kline_to_arrays(kline)
            mom = compute_factor_series("momentum", arr)
            rsi = compute_factor_series("rsi", arr)
            i = len(kline) - 1
            m = series_at(mom, i)
            r = series_at(rsi, i)
            tag = None
            if r is not None and r < 30:
                tag = "超卖"
            elif r is not None and r > 70:
                tag = "超买"
            if m is not None and m > 0.1:
                tag = (tag + "+突破" if tag else "突破")
            signals.append({"code": code, "momentum": m, "rsi": r, "signal": tag})
        _last_signals = signals
    except Exception:
        _last_signals = []
