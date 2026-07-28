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
