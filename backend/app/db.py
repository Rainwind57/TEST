"""SQLite 持久化层：自选股、模拟盘（现金/持仓/成交/净值曲线）。"""
import sqlite3
import os
import datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "quant.db")
INIT_CASH = 1_000_000.0

DEFAULT_CODES = ["sh600519", "sz000001", "sz300750", "sh601318"]

# 不可交易代码过滤：仅排除指数（sh000xxx/sz399xxx）和 ETF（sh51xxxx/sh56xxxx/sh58xxxx/sz15xxxx）
import re as _re
_INDEX_PATTERN = _re.compile(r"^(sh000|sz399)\d{3,4}$")
# ETF：沪市 51/56/58 开头 6 位，深市 15 开头 6 位
_ETF_PATTERN = _re.compile(r"^(sh5[168]|sz15)\d{4}$")


def is_tradable(code: str) -> bool:
    """校验代码是否为可交易个股（排除指数、ETF等不可交易品种）。"""
    c = code.strip().lower()
    if _INDEX_PATTERN.match(c):
        return False
    if _ETF_PATTERN.match(c):
        return False
    return True


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # WAL 模式：读不阻塞写、写不阻塞读，显著提升并发；synchronous=NORMAL 在 WAL 下安全且更快
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


@contextmanager
def get_conn():
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS watchlist (
            code TEXT PRIMARY KEY,
            name TEXT DEFAULT '',
            added_at TEXT
        )""")
        # 兼容旧表缺少 name 列
        try:
            cur.execute("ALTER TABLE watchlist ADD COLUMN name TEXT DEFAULT ''")
        except Exception:
            pass
        cur.execute("""CREATE TABLE IF NOT EXISTS portfolio_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            cash REAL NOT NULL
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS positions (
            code TEXT PRIMARY KEY,
            name TEXT,
            qty INTEGER NOT NULL,
            avg_cost REAL NOT NULL
        )""")
        # 做空支持：side=long|short（旧库兼容，默认 long）
        _ensure_column(cur, "positions", "side", "TEXT DEFAULT 'long'")
        cur.execute("""CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            side TEXT,
            code TEXT,
            name TEXT,
            qty INTEGER,
            price REAL,
            amount REAL
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS equity_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            value REAL
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS kline_cache (
            code TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL, close REAL, high REAL, low REAL, volume REAL,
            PRIMARY KEY (code, date)
        )""")
        _ensure_column(cur, "kline_cache", "adjust", "TEXT DEFAULT ''")
        cur.execute("""CREATE TABLE IF NOT EXISTS saved_strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            config TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS user_factors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            definition TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id INTEGER,
            config TEXT NOT NULL,
            metrics TEXT NOT NULL,
            report_path TEXT,
            created_at TEXT NOT NULL
        )""")
        # P10：回测结果全量落盘（含 longShort/groupSummary/icSeries），供报告历史页"一键重生成"
        _ensure_column(cur, "backtest_runs", "result_json", "TEXT")
        cur.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        # 异步任务表：旧版 job 元数据纯内存，重启即丢、多 worker 不共享。落 SQLite 后历史可查。
        cur.execute("""CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            config TEXT NOT NULL,
            status TEXT NOT NULL,
            progress INTEGER DEFAULT 0,
            message TEXT DEFAULT '',
            result TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            finished_at TEXT
        )""")
        _ensure_column(cur, "jobs", "user_id", "INTEGER DEFAULT 0")
        # 调度器运行记录：旧版 _last_run/_last_signals 纯内存，重启即丢、无法审计。
        cur.execute("""CREATE TABLE IF NOT EXISTS scheduler_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT NOT NULL,
            ts TEXT NOT NULL,
            success INTEGER NOT NULL,
            payload TEXT,
            error TEXT
        )""")
        # P0修复：kv 配置表，用于持久化调度器开关等状态（重启后恢复）
        cur.execute("""CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )""")

        # 多用户：为研究资产表加 user_id 列（ALTER 兼容旧库）
        _ensure_column(cur, "saved_strategies", "user_id", "INTEGER DEFAULT 0")
        _ensure_column(cur, "user_factors", "user_id", "INTEGER DEFAULT 0")
        _ensure_column(cur, "backtest_runs", "user_id", "INTEGER DEFAULT 0")

        cur.execute("SELECT COUNT(*) AS c FROM watchlist")
        if cur.fetchone()["c"] == 0:
            now = datetime.datetime.now().isoformat()
            for code in DEFAULT_CODES:
                cur.execute("INSERT INTO watchlist (code, added_at) VALUES (?, ?)", (code, now))

        cur.execute("SELECT COUNT(*) AS c FROM portfolio_state")
        if cur.fetchone()["c"] == 0:
            cur.execute("INSERT INTO portfolio_state (id, cash) VALUES (1, ?)", (INIT_CASH,))
            cur.execute("INSERT INTO equity_history (ts, value) VALUES (?, ?)",
                        (datetime.datetime.now().isoformat(), INIT_CASH))

        conn.commit()

    # 迁移：分配策略旧默认 0.2/5 → 新默认 0.1/20（一次性，防止旧库持久化值覆盖新代码默认）
    _migrate_alloc_defaults()


def _migrate_alloc_defaults() -> None:
    """批量买入分配默认值迁移（一次性）：旧默认 perPositionPct=0.2 / maxPositions=5
    会被 set_signal_config 持久化到 settings 表，若不迁移将一直覆盖新代码默认 0.1/20。"""
    if get_setting("alloc_defaults_migrated_v2") == "1":
        return
    if get_setting("monitor_alloc_per_pos_pct", "0.1") == "0.2":
        set_setting("monitor_alloc_per_pos_pct", "0.1")
    if get_setting("monitor_alloc_max_positions", "20") == "5":
        set_setting("monitor_alloc_max_positions", "20")
    set_setting("alloc_defaults_migrated_v2", "1")


def reset_portfolio():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE portfolio_state SET cash = ? WHERE id = 1", (INIT_CASH,))
        cur.execute("DELETE FROM positions")
        cur.execute("DELETE FROM trades")
        cur.execute("DELETE FROM equity_history")
        cur.execute("INSERT INTO equity_history (ts, value) VALUES (?, ?)",
                    (datetime.datetime.now().isoformat(), INIT_CASH))
        conn.commit()


# ---------------- K线磁盘缓存 ----------------

def get_cached_kline(code: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT date, open, close, high, low, volume, adjust FROM kline_cache WHERE code = ? ORDER BY date",
            (code,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_cached_kline_adjust(code: str) -> str:
    """返回缓存中该股票的复权标记，空字符串表示无标记或旧数据。"""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT adjust FROM kline_cache WHERE code = ? LIMIT 1", (code,)
        ).fetchone()
        return row["adjust"] if row else ""


def upsert_kline(code: str, rows: list[dict], adjust: str = ""):
    if not rows:
        return
    with get_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO kline_cache (code, date, open, close, high, low, volume, adjust) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(code, r["date"], r["open"], r["close"], r["high"], r["low"], r["volume"], adjust) for r in rows]
        )
        conn.commit()


def clear_kline_cache(code: str):
    """清除单只股票的K线缓存（复权基准漂移时整表重写前调用）。"""
    with get_conn() as conn:
        conn.execute("DELETE FROM kline_cache WHERE code = ?", (code,))
        conn.commit()


# ---------------- 策略 / 自定义因子 / 回测存档 CRUD ----------------

def list_strategies(user_id: int = 0) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM saved_strategies WHERE user_id = ? ORDER BY id DESC", (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def create_strategy(name: str, kind: str, config: dict, user_id: int = 0) -> dict:
    with get_conn() as conn:
        now = datetime.datetime.now().isoformat()
        cur = conn.execute(
            "INSERT INTO saved_strategies (name, kind, config, created_at, user_id) VALUES (?, ?, ?, ?, ?)",
            (name, kind, _json_dumps(config), now, user_id)
        )
        conn.commit()
        row = conn.execute("SELECT * FROM saved_strategies WHERE id = ?", (cur.lastrowid,)).fetchone()
        return _parse_strategy(row)


def delete_strategy(strategy_id: int, user_id: int | None = None) -> bool:
    with get_conn() as conn:
        if user_id is None:
            cur = conn.execute("DELETE FROM saved_strategies WHERE id = ?", (strategy_id,))
        else:
            cur = conn.execute(
                "DELETE FROM saved_strategies WHERE id = ? AND user_id = ?",
                (strategy_id, user_id))
        conn.commit()
        return cur.rowcount > 0


def get_strategy(strategy_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM saved_strategies WHERE id = ?", (strategy_id,)).fetchone()
        return _parse_strategy(row) if row else None


def list_user_factors(user_id: int = 0) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM user_factors WHERE user_id = ? ORDER BY id DESC", (user_id,)
        ).fetchall()
        return [_parse_user_factor(r) for r in rows]


def create_user_factor(name: str, kind: str, definition: dict, user_id: int = 0) -> dict:
    with get_conn() as conn:
        now = datetime.datetime.now().isoformat()
        cur = conn.execute(
            "INSERT INTO user_factors (name, kind, definition, created_at, user_id) VALUES (?, ?, ?, ?, ?)",
            (name, kind, _json_dumps(definition), now, user_id)
        )
        conn.commit()
        row = conn.execute("SELECT * FROM user_factors WHERE id = ?", (cur.lastrowid,)).fetchone()
        return _parse_user_factor(row)


def delete_user_factor(factor_id: int, user_id: int | None = None) -> bool:
    with get_conn() as conn:
        if user_id is None:
            cur = conn.execute("DELETE FROM user_factors WHERE id = ?", (factor_id,))
        else:
            cur = conn.execute(
                "DELETE FROM user_factors WHERE id = ? AND user_id = ?",
                (factor_id, user_id))
        conn.commit()
        return cur.rowcount > 0


def get_user_factor(factor_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM user_factors WHERE id = ?", (factor_id,)).fetchone()
        return _parse_user_factor(row) if row else None


def list_backtest_runs(limit: int = 50, user_id: int | None = 0) -> list[dict]:
    with get_conn() as conn:
        if user_id is None:
            # 报告历史不按用户过滤（单机 SQLite，回测路由未强制登录时存档为 user_id=0）
            rows = conn.execute(
                "SELECT * FROM backtest_runs ORDER BY id DESC LIMIT ?",
                (max(1, min(limit, 200)),)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM backtest_runs WHERE user_id = ? OR user_id = 0 ORDER BY id DESC LIMIT ?",
                (user_id, max(1, min(limit, 200)))
            ).fetchall()
        return [_parse_backtest_run(r) for r in rows]


def create_backtest_run(strategy_id, config: dict, metrics: dict, report_path=None,
                        user_id: int = 0, result=None) -> dict:
    with get_conn() as conn:
        now = datetime.datetime.now().isoformat()
        cur = conn.execute(
            "INSERT INTO backtest_runs (strategy_id, config, metrics, report_path, result_json, created_at, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (strategy_id, _json_dumps(config), _json_dumps(metrics), report_path,
             _json_dumps(result) if result is not None else None, now, user_id)
        )
        conn.commit()
        row = conn.execute("SELECT * FROM backtest_runs WHERE id = ?", (cur.lastrowid,)).fetchone()
        return _parse_backtest_run(row)


def delete_backtest_run(run_id: int, user_id: int | None = None) -> bool:
    with get_conn() as conn:
        if user_id is None:
            cur = conn.execute("DELETE FROM backtest_runs WHERE id = ?", (run_id,))
        else:
            cur = conn.execute(
                "DELETE FROM backtest_runs WHERE id = ? AND user_id = ?",
                (run_id, user_id))
        conn.commit()
        return cur.rowcount > 0


def _json_dumps(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)


def _ensure_column(cur, table: str, column: str, definition: str):
    """为已存在的表添加列（SQLite 无 IF NOT EXISTS for ADD COLUMN），忽略重复列错误。"""
    cols = {r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _parse_strategy(row):
    import json
    if not row:
        return None
    d = dict(row)
    try:
        d["config"] = json.loads(d["config"]) if d.get("config") else {}
    except (json.JSONDecodeError, TypeError):
        pass
    return d


def _parse_user_factor(row):
    import json
    if not row:
        return None
    d = dict(row)
    try:
        d["definition"] = json.loads(d["definition"]) if d.get("definition") else {}
    except (json.JSONDecodeError, TypeError):
        pass
    return d


def _parse_backtest_run(row):
    import json
    if not row:
        return None
    d = dict(row)
    try:
        d["config"] = json.loads(d["config"]) if d.get("config") else {}
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        d["metrics"] = json.loads(d["metrics"]) if d.get("metrics") else {}
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        d["result"] = json.loads(d["result_json"]) if d.get("result_json") else None
    except (json.JSONDecodeError, TypeError):
        d["result"] = None
    return d


# ---------------- 异步任务（jobs 表） ----------------

def create_job_row(jid: str, kind: str, config: dict, user_id: int = 0) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO jobs (id, kind, config, status, progress, message, created_at, user_id) "
            "VALUES (?, ?, ?, 'pending', 0, '', ?, ?)",
            (jid, kind, _json_dumps(config), datetime.datetime.now().isoformat(), user_id)
        )
        conn.commit()


def get_job_row(jid: str, user_id: int | None = None) -> dict | None:
    with get_conn() as conn:
        if user_id is not None:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ? AND user_id = ?", (jid, user_id)
            ).fetchone()
        else:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (jid,)).fetchone()
        return _parse_job_row(row) if row else None


def list_job_rows(limit: int = 50, user_id: int | None = None) -> list[dict]:
    with get_conn() as conn:
        if user_id is not None:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, max(1, min(limit, 200)))
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 200)),)
            ).fetchall()
        return [_parse_job_row(r) for r in rows]


def update_job_row(jid: str, **fields) -> None:
    allowed = {"status", "progress", "message", "result", "error", "finished_at"}
    sets = []
    vals = []
    for k, v in fields.items():
        if k not in allowed:
            continue
        if k == "result":
            v = _json_dumps(v) if v is not None else None
        sets.append(f"{k} = ?")
        vals.append(v)
    if not sets:
        return
    vals.append(jid)
    with get_conn() as conn:
        conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", vals)
        conn.commit()


def delete_job_row(jid: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM jobs WHERE id = ?", (jid,))
        conn.commit()


def prune_old_jobs(max_rows: int = 200) -> None:
    """超过上限时删除最旧的已终结 job（done/error/cancelled）。"""
    with get_conn() as conn:
        finished = conn.execute(
            "SELECT id FROM jobs WHERE status IN ('done','error','cancelled') "
            "ORDER BY created_at ASC"
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) AS c FROM jobs").fetchone()["c"]
        if total > max_rows:
            for r in finished[:total - max_rows]:
                conn.execute("DELETE FROM jobs WHERE id = ?", (r["id"],))
        conn.commit()


def _parse_job_row(row):
    import json
    if not row:
        return None
    d = dict(row)
    try:
        d["config"] = json.loads(d["config"]) if d.get("config") else {}
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        d["result"] = json.loads(d["result"]) if d.get("result") else None
    except (json.JSONDecodeError, TypeError):
        pass
    return d


# ---------------- 调度器运行记录（scheduler_runs 表） ----------------

def log_scheduler_run(task: str, success: bool, payload: dict | None = None,
                      error: str | None = None) -> None:
    """写一条调度器运行记录（_last_run/_last_signals 的持久化镜像）。"""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO scheduler_runs (task, ts, success, payload, error) "
            "VALUES (?, ?, ?, ?, ?)",
            (task, datetime.datetime.now().isoformat(),
             1 if success else 0,
             _json_dumps(payload) if payload is not None else None,
             error)
        )
        conn.commit()


def list_scheduler_runs(limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, task, ts, success, payload, error FROM scheduler_runs "
            "ORDER BY id DESC LIMIT ?",
            (max(1, min(limit, 100)),)
        ).fetchall()
        import json
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["payload"] = json.loads(d["payload"]) if d.get("payload") else None
            except (json.JSONDecodeError, TypeError):
                pass
            out.append(d)
        return out


def get_last_scheduler_run(task: str) -> dict | None:
    """取某个任务最近一次运行记录（供 monitor 接口在内存态丢失后兜底）。"""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, task, ts, success, payload, error FROM scheduler_runs "
            "WHERE task = ? ORDER BY id DESC LIMIT 1",
            (task,)
        ).fetchone()
        if not row:
            return None
        import json
        d = dict(row)
        try:
            d["payload"] = json.loads(d["payload"]) if d.get("payload") else None
        except (json.JSONDecodeError, TypeError):
            pass
        return d


# ---------------- settings 持久化（P0修复） ----------------

def get_setting(key: str, default: str = "") -> str:
    with get_conn() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with get_conn() as conn:
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
        conn.commit()


def get_scheduler_enabled() -> bool:
    return get_setting("scheduler_enabled", "0") == "1"


def set_scheduler_enabled(enabled: bool) -> None:
    set_setting("scheduler_enabled", "1" if enabled else "0")
