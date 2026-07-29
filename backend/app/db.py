"""SQLite 持久化层：自选股、模拟盘（现金/持仓/成交/净值曲线）。"""
import sqlite3
import os
import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "quant.db")
INIT_CASH = 1_000_000.0

DEFAULT_CODES = ["sh000001", "sh600519", "sz000001", "sz300750", "sh601318"]


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS watchlist (
        code TEXT PRIMARY KEY,
        added_at TEXT
    )""")
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
    conn.close()


def reset_portfolio():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE portfolio_state SET cash = ? WHERE id = 1", (INIT_CASH,))
    cur.execute("DELETE FROM positions")
    cur.execute("DELETE FROM trades")
    cur.execute("DELETE FROM equity_history")
    cur.execute("INSERT INTO equity_history (ts, value) VALUES (?, ?)",
                (datetime.datetime.now().isoformat(), INIT_CASH))
    conn.commit()
    conn.close()


# ---------------- K线磁盘缓存 ----------------

def get_cached_kline(code: str) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT date, open, close, high, low, volume FROM kline_cache WHERE code = ? ORDER BY date",
        (code,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def upsert_kline(code: str, rows: list[dict]):
    if not rows:
        return
    conn = get_conn()
    conn.executemany(
        "INSERT OR REPLACE INTO kline_cache (code, date, open, close, high, low, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(code, r["date"], r["open"], r["close"], r["high"], r["low"], r["volume"]) for r in rows]
    )
    conn.commit()
    conn.close()


# ---------------- 策略 / 自定义因子 / 回测存档 CRUD ----------------

def list_strategies() -> list[dict]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM saved_strategies ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_strategy(name: str, kind: str, config: dict) -> dict:
    conn = get_conn()
    now = datetime.datetime.now().isoformat()
    cur = conn.execute(
        "INSERT INTO saved_strategies (name, kind, config, created_at) VALUES (?, ?, ?, ?)",
        (name, kind, _json_dumps(config), now)
    )
    conn.commit()
    row = conn.execute("SELECT * FROM saved_strategies WHERE id = ?", (cur.lastrowid,)).fetchone()
    conn.close()
    return _parse_strategy(row)


def delete_strategy(strategy_id: int) -> bool:
    conn = get_conn()
    cur = conn.execute("DELETE FROM saved_strategies WHERE id = ?", (strategy_id,))
    conn.commit()
    conn.close()
    return cur.rowcount > 0


def get_strategy(strategy_id: int):
    conn = get_conn()
    row = conn.execute("SELECT * FROM saved_strategies WHERE id = ?", (strategy_id,)).fetchone()
    conn.close()
    return _parse_strategy(row) if row else None


def list_user_factors() -> list[dict]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM user_factors ORDER BY id DESC").fetchall()
    conn.close()
    return [_parse_user_factor(r) for r in rows]


def create_user_factor(name: str, kind: str, definition: dict) -> dict:
    conn = get_conn()
    now = datetime.datetime.now().isoformat()
    cur = conn.execute(
        "INSERT INTO user_factors (name, kind, definition, created_at) VALUES (?, ?, ?, ?)",
        (name, kind, _json_dumps(definition), now)
    )
    conn.commit()
    row = conn.execute("SELECT * FROM user_factors WHERE id = ?", (cur.lastrowid,)).fetchone()
    conn.close()
    return _parse_user_factor(row)


def delete_user_factor(factor_id: int) -> bool:
    conn = get_conn()
    cur = conn.execute("DELETE FROM user_factors WHERE id = ?", (factor_id,))
    conn.commit()
    conn.close()
    return cur.rowcount > 0


def get_user_factor(factor_id: int):
    conn = get_conn()
    row = conn.execute("SELECT * FROM user_factors WHERE id = ?", (factor_id,)).fetchone()
    conn.close()
    return _parse_user_factor(row) if row else None


def list_backtest_runs(limit: int = 50) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM backtest_runs ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [_parse_backtest_run(r) for r in rows]


def create_backtest_run(strategy_id, config: dict, metrics: dict, report_path=None) -> dict:
    conn = get_conn()
    now = datetime.datetime.now().isoformat()
    cur = conn.execute(
        "INSERT INTO backtest_runs (strategy_id, config, metrics, report_path, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (strategy_id, _json_dumps(config), _json_dumps(metrics), report_path, now)
    )
    conn.commit()
    row = conn.execute("SELECT * FROM backtest_runs WHERE id = ?", (cur.lastrowid,)).fetchone()
    conn.close()
    return _parse_backtest_run(row)


def delete_backtest_run(run_id: int) -> bool:
    conn = get_conn()
    cur = conn.execute("DELETE FROM backtest_runs WHERE id = ?", (run_id,))
    conn.commit()
    conn.close()
    return cur.rowcount > 0


def _json_dumps(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)


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
    return d
