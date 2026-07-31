"""异步任务队列：长回测改「提交 → 返回 job_id → 轮询」。

job 元数据持久化到 SQLite（jobs 表），重启后历史可查、多 worker 可共享；
asyncio Task 句柄仅存内存（重启即丢，asyncio 固有限制，对已入库的终态记录无影响）。
"""
import asyncio
import datetime

from . import db

_tasks: dict[str, asyncio.Task] = {}
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


def _runner(jid: str, coro):
    async def wrapper():
        try:
            update_job(jid, status="running", message="任务执行中")
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
    _tasks[jid] = asyncio.create_task(_runner(jid, coro)())


def cancel(jid: str) -> bool:
    t = _tasks.get(jid)
    if t and not t.done():
        t.cancel()
        update_job(jid, status="cancelled", finished_at=datetime.datetime.now().isoformat())
        return True
    return False
