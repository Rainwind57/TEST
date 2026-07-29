"""异步任务队列：长回测改「提交 → 返回 job_id → 轮询」。

内存 job 表（进程重启即丢失，足够单机研究用）。回测在 asyncio 后台 Task 中执行，
主请求立即返回 job_id，前端轮询 /api/jobs/{id} 获取进度与结果。
"""
import asyncio
import datetime
import uuid

_jobs: dict[str, dict] = {}
_tasks: dict[str, asyncio.Task] = {}


def create_job(kind: str, config: dict) -> str:
    jid = uuid.uuid4().hex[:12]
    _jobs[jid] = {
        "id": jid,
        "kind": kind,
        "config": config,
        "status": "pending",
        "progress": 0,
        "message": "",
        "result": None,
        "error": None,
        "created_at": datetime.datetime.now().isoformat(),
        "finished_at": None,
    }
    return jid


def get_job(jid: str) -> dict | None:
    return _jobs.get(jid)


def list_jobs(limit: int = 50) -> list[dict]:
    rows = sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)
    return rows[:limit]


def update_job(jid: str, **fields):
    j = _jobs.get(jid)
    if j:
        j.update(fields)


def _runner(jid: str, coro):
    async def wrapper():
        try:
            update_job(jid, status="running", message="任务执行中")
            result = await coro
            update_job(jid, status="done", progress=100, result=result,
                       finished_at=datetime.datetime.now().isoformat())
        except Exception as e:
            update_job(jid, status="error", error=str(e),
                       finished_at=datetime.datetime.now().isoformat())
    return wrapper


def submit(jid: str, coro):
    _tasks[jid] = asyncio.create_task(_runner(jid, coro)())


def cancel(jid: str) -> bool:
    t = _tasks.get(jid)
    if t and not t.done():
        t.cancel()
        update_job(jid, status="cancelled", finished_at=datetime.datetime.now().isoformat())
        return True
    return False
