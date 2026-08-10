"""异步任务队列：长回测改「提交 → 返回 job_id → 轮询」。

job 元数据持久化到 SQLite（jobs 表），重启后历史可查、多 worker 可共享；
asyncio Task 句柄仅存内存（重启即丢，asyncio 固有限制，对已入库的终态记录无影响）。

取消机制：cancel() 同时触发 Task.cancel()（向 coro 投递 CancelledError）
和置 cancel_ev。coro 应在长时间循环中调用 is_cancelled(jid) 轮询，
避免长时间同步阻塞段无法及时响应取消。
"""
import asyncio
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

from . import db

_tasks: dict[str, asyncio.Task] = {}
_cancel_events: dict[str, threading.Event] = {}
# 独立线程池，避免耗尽 FastAPI 默认线程池导致整站假死
_JOB_EXECUTOR = ThreadPoolExecutor(max_workers=2)
MAX_JOBS = 200  # 库内 job 上限，防无限增长


def create_job(kind: str, config: dict) -> str:
    import uuid
    jid = uuid.uuid4().hex[:12]
    db.create_job_row(jid, kind, config)
    return jid


def get_job(jid: str) -> dict | None:
    return db.get_job_row(jid)


def list_jobs(limit: int = 50) -> list[dict]:
    return db.list_job_rows(limit)


def update_job(jid: str, **fields):
    db.update_job_row(jid, **fields)
    if fields.get("status") in ("done", "error", "cancelled"):
        db.prune_old_jobs(MAX_JOBS)


def is_cancelled(jid: str) -> bool:
    """长任务在循环中调用，检查是否被取消。

    返回 True 时应主动 raise asyncio.CancelledError() 或 break。
    """
    ev = _cancel_events.get(jid)
    return bool(ev and ev.is_set())


def _runner(jid: str, coro, cancel_ev: threading.Event | None = None):
    async def wrapper():
        try:
            update_job(jid, status="running", message="任务执行中")
            if cancel_ev and cancel_ev.is_set():
                raise asyncio.CancelledError()
            result = await coro
            update_job(jid, status="done", progress=100, result=result,
                       finished_at=datetime.datetime.now().isoformat())
        except asyncio.CancelledError:
            update_job(jid, status="cancelled",
                       finished_at=datetime.datetime.now().isoformat())
            raise
        except Exception as e:
            update_job(jid, status="error", error=str(e),
                       finished_at=datetime.datetime.now().isoformat())
    return wrapper


def submit(jid: str, coro):
    db.prune_old_jobs(MAX_JOBS)
    event = threading.Event()
    _cancel_events[jid] = event
    _tasks[jid] = asyncio.create_task(_runner(jid, coro, event)())


def cancel(jid: str) -> bool:
    t = _tasks.get(jid)
    if t and not t.done():
        t.cancel()
        ev = _cancel_events.pop(jid, None)
        if ev:
            ev.set()
        update_job(jid, status="cancelled", finished_at=datetime.datetime.now().isoformat())
        return True
    return False


def get_executor() -> ThreadPoolExecutor:
    return _JOB_EXECUTOR


def get_cancel_event(jid: str) -> threading.Event | None:
    return _cancel_events.get(jid)
