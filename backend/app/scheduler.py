"""定时盯盘调度器：APScheduler 后台任务。

功能：
- 交易日 15:05 自动刷新持仓市值并写 equity_history（日频净值完整）
- 交易日 15:10 重算 watchlist 因子打分，生成信号（可选自动调仓，默认关闭）

默认关闭，需用户显式开启（模拟边界）。内存状态仅作实时返回，关键运行记录
（_last_run / 信号结果）已落 scheduler_runs 表，重启后可审计、可回溯。
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

# 交易日历缓存：用上证指数 K 线日期集合推断交易日，替代硬编码 _HOLIDAYS。
# 有 K 线的日期必为交易日，无需维护年度节假日表，也无 2027 失效问题。
_TRADING_DAYS: set[str] | None = None
_TRADING_DAYS_UPDATED: float = 0.0
_TRADING_DAYS_TTL = 24 * 3600  # 一天最多回源一次

# 盯盘信号单股拉取并发上限（取代旧版 codes[:20] 硬截断，全量扫描但限流防刷接口）
_SCAN_CONCURRENCY = 15


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
    db.set_scheduler_enabled(True)


def stop():
    global _scheduler, _enabled
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
    _enabled = False
    db.set_scheduler_enabled(False)


async def _load_trading_days() -> set[str]:
    """取上证指数 K 线日期集合作为交易日历（带 24h 内存缓存）。

    替代旧版硬编码 _HOLIDAYS（仅 2026 年、未含调休），避免失效及
    调休误判。回源失败时退化为「非周末即交易日」的近似，保证调度不中断。
    """
    global _TRADING_DAYS, _TRADING_DAYS_UPDATED
    import time
    now = time.time()
    if _TRADING_DAYS is not None and now - _TRADING_DAYS_UPDATED < _TRADING_DAYS_TTL:
        return _TRADING_DAYS
    try:
        # 取足够长的历史以覆盖近期所有交易日（含节假日前后）
        kline = await adapters.fetch_kline("sh000001", 400)
        days = {row["date"] for row in kline if row.get("date")}
        if days:
            _TRADING_DAYS = days
            _TRADING_DAYS_UPDATED = now
            return days
    except Exception:
        pass
    # 回源失败：退化近似（非周末即交易日），宁可多跑一次 refresh 也别漏跑
    return _TRADING_DAYS or set()


async def _is_trading_day() -> bool:
    """交易日判断：周末休市 + 交易日历校验（交易日历回源失败时仅判周末）。"""
    now = datetime.datetime.now()
    if now.weekday() >= 5:
        return False
    days = await _load_trading_days()
    if not days:
        return True  # 日历为空（回源失败）时退化为「非周末即交易日」
    return now.strftime("%Y-%m-%d") in days


async def _refresh_equity():
    if not await _is_trading_day():
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
        db.log_scheduler_run("refresh_equity", True, _last_run)
    except Exception as e:
        _last_run = {"task": "refresh_equity", "error": str(e)}
        db.log_scheduler_run("refresh_equity", False, error=str(e))


async def _scan_signals():
    if not await _is_trading_day():
        return
    global _last_signals
    try:
        conn = db.get_conn()
        codes = [r["code"] for r in conn.execute("SELECT code FROM watchlist ORDER BY added_at").fetchall()]
        conn.close()
        if not codes:
            _last_signals = []
            return
        # 取消旧版 codes[:20] 硬截断：全量扫描 + 信号量限流，超 20 只不再静默丢失
        sem = asyncio.Semaphore(_SCAN_CONCURRENCY)
        signals: list[dict] = []

        async def one(code: str):
            async with sem:
                try:
                    kline = await adapters.fetch_kline(code, 60)
                except Exception:
                    return
                if len(kline) < 25:
                    return
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

        await asyncio.gather(*(one(c) for c in codes))
        _last_signals = signals
        db.log_scheduler_run("scan_signals", True, {"count": len(signals)})
    except Exception as e:
        _last_signals = []
        db.log_scheduler_run("scan_signals", False, error=str(e))
