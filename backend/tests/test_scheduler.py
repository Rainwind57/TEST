"""scheduler.py 调度器单元测试（P2-13 + P0-3 回归）。

覆盖：
- _load_trading_days 返回 set 且非空（K 线日期集合）
- db.log_scheduler_run / list_scheduler_runs / get_last_scheduler_run 落库
- _scan_signals 取消 codes[:20] 硬截断（源码级回归）
"""
import inspect
import pytest

from app import scheduler, db


def test_scheduler_runs_table_created():
    """init_db 应建 scheduler_runs 表（P0-3c）。"""
    db.init_db()
    with db.get_conn() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(scheduler_runs)").fetchall()}
    assert {"task", "ts", "success", "payload", "error"} <= cols


def test_log_and_list_scheduler_run():
    """写入一条运行记录，list/get_last 能读到（P0-3c）。"""
    db.init_db()
    db.log_scheduler_run("test_task", True, {"key": "value"})
    runs = db.list_scheduler_runs(10)
    assert any(r["task"] == "test_task" for r in runs)
    last = db.get_last_scheduler_run("test_task")
    assert last is not None
    assert last["success"] == 1
    assert last["payload"] == {"key": "value"}


def test_log_scheduler_run_error():
    """失败记录也能落库（审计完整性）。"""
    db.init_db()
    db.log_scheduler_run("test_err", False, error="boom")
    last = db.get_last_scheduler_run("test_err")
    assert last is not None
    assert last["success"] == 0
    assert last["error"] == "boom"


@pytest.mark.asyncio
async def test_load_trading_days_returns_set():
    """_load_trading_days 返回 set（P0-3a，替代硬编码 _HOLIDAYS）。"""
    # 模块不应再定义 _HOLIDAYS 集合变量（注释提及不算）
    assert not hasattr(scheduler, "_HOLIDAYS")
    # 取消硬编码后应有 _TRADING_DAYS 缓存与 _load_trading_days 函数
    assert hasattr(scheduler, "_TRADING_DAYS")
    assert inspect.iscoroutinefunction(scheduler._load_trading_days)
    days = await scheduler._load_trading_days()
    assert isinstance(days, set)


def test_scan_signals_no_20_hardcap():
    """_scan_signals_impl 不再硬截断 codes[:20]（P0-3b 回归）。

    检查函数体 AST，不存在对 codes 的 [:20] 切片；应改用信号量限流。
    """
    import ast
    src = inspect.getsource(scheduler._scan_signals_impl)
    tree = ast.parse(src)
    found_hardcap = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            # codes[:20] 形式：Subscript(value=Name('codes'), slice=Slice(lower=None, upper=Num(20)))
            if (isinstance(node.value, ast.Name) and node.value.id == "codes"
                and node.slice.upper is not None):
                try:
                    upper_val = ast.literal_eval(node.slice.upper)
                except Exception:
                    upper_val = None
                if upper_val == 20:
                    found_hardcap = True
    assert not found_hardcap, "_scan_signals_impl 仍含 codes[:20] 硬截断"
    # 应改用信号量限流
    assert "Semaphore" in src


def test_scan_now_force_runs_impl(monkeypatch):
    """scan_now(force=True) 应跳过交易日判断并执行一次扫描（返回 ok + signals）。"""
    import asyncio

    calls = {"n": 0}

    async def fake_impl():
        calls["n"] += 1
        scheduler._last_signals = []

    monkeypatch.setattr(scheduler, "_scan_signals_impl", fake_impl)
    result = asyncio.run(scheduler.scan_now(force=True))
    assert result["ok"] is True
    assert isinstance(result["signals"], list)
    assert calls["n"] == 1


def test_scan_now_skips_non_trading_day(monkeypatch):
    """未加 force 且非交易日时应跳过（ok=False 带原因），不执行扫描。"""
    import asyncio

    async def fake_is_trading_day():
        return False

    async def fake_impl():
        raise AssertionError("非交易日不应执行扫描")

    monkeypatch.setattr(scheduler, "_is_trading_day", fake_is_trading_day)
    monkeypatch.setattr(scheduler, "_scan_signals_impl", fake_impl)
    result = asyncio.run(scheduler.scan_now(force=False))
    assert result["ok"] is False
    assert "非交易日" in result["reason"]


def test_set_signal_config_rejects_missing_model(tmp_path, monkeypatch):
    """模型模式下 modelId 不存在应拒绝保存（P2：防配置残留失效模型导致静默失败）。"""
    settings = {}
    monkeypatch.setattr(scheduler.db, "set_setting", lambda k, v: settings.__setitem__(k, v))
    monkeypatch.setattr(scheduler.db, "get_setting", lambda k, d="": settings.get(k, d))
    from app import ml
    monkeypatch.setattr(ml, "ML_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="模型不存在"):
        scheduler.set_signal_config("model", "no_such_model")

    # 模型文件存在时允许保存
    (tmp_path / "mlmodel_ok.joblib").write_bytes(b"x")
    cfg = scheduler.set_signal_config("model", "mlmodel_ok")
    assert cfg["modelId"] == "mlmodel_ok"


def test_set_signal_config_model_requires_id(monkeypatch):
    """模型模式必须指定 modelId。"""
    settings = {}
    monkeypatch.setattr(scheduler.db, "set_setting", lambda k, v: settings.__setitem__(k, v))
    monkeypatch.setattr(scheduler.db, "get_setting", lambda k, d="": settings.get(k, d))
    with pytest.raises(ValueError, match="modelId"):
        scheduler.set_signal_config("model", "")
