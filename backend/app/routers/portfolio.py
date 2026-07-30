"""模拟盘路由：下单、持仓查询、净值曲线、重置。"""
import datetime
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .. import adapters, db
from ..auth import get_user_id_from_auth

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


class OrderBody(BaseModel):
    code: str
    side: str  # "buy" | "sell"
    qty: int


async def _build_portfolio_view():
    conn = db.get_conn()
    cash = conn.execute("SELECT cash FROM portfolio_state WHERE id = 1").fetchone()["cash"]
    positions = conn.execute("SELECT code, name, qty, avg_cost FROM positions").fetchall()
    trades = conn.execute(
        "SELECT time, side, code, name, qty, price, amount FROM trades ORDER BY id DESC LIMIT 30"
    ).fetchall()
    equity_rows = conn.execute(
        "SELECT ts, value FROM equity_history ORDER BY id DESC LIMIT 300"
    ).fetchall()
    conn.close()

    codes = [p["code"] for p in positions]
    quotes = {}
    if codes:
        try:
            quotes = await adapters.fetch_tencent_quotes(codes)
        except Exception:
            quotes = {}

    pos_list = []
    market_value = 0.0
    for p in positions:
        q = quotes.get(p["code"])
        price = q["price"] if q else p["avg_cost"]
        value = price * p["qty"]
        market_value += value
        pnl = (price - p["avg_cost"]) * p["qty"]
        pnl_pct = (price / p["avg_cost"] - 1) * 100 if p["avg_cost"] else 0.0
        pos_list.append({
            "code": p["code"], "name": p["name"], "qty": p["qty"],
            "avgCost": p["avg_cost"], "price": price, "marketValue": value,
            "pnl": pnl, "pnlPct": pnl_pct,
        })

    total_assets = cash + market_value
    total_pnl = total_assets - db.INIT_CASH
    total_pnl_pct = (total_pnl / db.INIT_CASH) * 100

    return {
        "cash": cash,
        "marketValue": market_value,
        "totalAssets": total_assets,
        "totalPnl": total_pnl,
        "totalPnlPct": total_pnl_pct,
        "positions": pos_list,
        "trades": [dict(t) for t in trades],
        "equity": [dict(e) for e in reversed(equity_rows)],
    }


@router.get("")
async def get_portfolio():
    return await _build_portfolio_view()


@router.post("/order")
async def place_order(body: OrderBody):
    if body.side not in ("buy", "sell"):
        raise HTTPException(400, "side 必须为 buy 或 sell")
    if body.qty <= 0 or body.qty % 100 != 0:
        raise HTTPException(400, "数量需为 100 的整数倍")

    code = body.code.strip().lower()
    try:
        quotes = await adapters.fetch_tencent_quotes([code])
    except Exception as e:
        raise HTTPException(502, f"行情获取失败: {e}")
    q = quotes.get(code)
    if not q or not q.get("price"):
        raise HTTPException(422, "暂无行情数据，请稍后重试")

    price = q["price"]
    amount = price * body.qty
    conn = db.get_conn()
    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE")  # 获取写锁，防并发下单双花（旧版读改写非原子）
    cash = cur.execute("SELECT cash FROM portfolio_state WHERE id = 1").fetchone()["cash"]
    pos = cur.execute("SELECT * FROM positions WHERE code = ?", (code,)).fetchone()

    if body.side == "buy":
        if amount > cash:
            conn.close()
            raise HTTPException(422, "资金不足")
        new_cash = cash - amount
        if pos:
            new_qty = pos["qty"] + body.qty
            new_avg_cost = (pos["avg_cost"] * pos["qty"] + amount) / new_qty
            cur.execute("UPDATE positions SET qty = ?, avg_cost = ?, name = ? WHERE code = ?",
                        (new_qty, new_avg_cost, q["name"], code))
        else:
            cur.execute("INSERT INTO positions (code, name, qty, avg_cost) VALUES (?, ?, ?, ?)",
                        (code, q["name"], body.qty, price))
    else:
        if not pos or pos["qty"] < body.qty:
            conn.close()
            raise HTTPException(422, "持仓不足")
        new_cash = cash + amount
        remain = pos["qty"] - body.qty
        if remain <= 0:
            cur.execute("DELETE FROM positions WHERE code = ?", (code,))
        else:
            cur.execute("UPDATE positions SET qty = ? WHERE code = ?", (remain, code))

    cur.execute("UPDATE portfolio_state SET cash = ? WHERE id = 1", (new_cash,))
    cur.execute(
        "INSERT INTO trades (time, side, code, name, qty, price, amount) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), body.side, code, q["name"], body.qty, price, amount)
    )
    conn.commit()
    conn.close()

    view = await _build_portfolio_view()
    _record_equity(view["totalAssets"])
    return view


def _record_equity(value: float):
    conn = db.get_conn()
    conn.execute("INSERT INTO equity_history (ts, value) VALUES (?, ?)",
                 (datetime.datetime.now().isoformat(), value))
    conn.commit()
    conn.close()


@router.post("/reset")
async def reset_portfolio(request: Request):
    if not get_user_id_from_auth(request.headers.get("Authorization")):
        raise HTTPException(401, "请先登录后再重置模拟盘")
    db.reset_portfolio()
    return await _build_portfolio_view()
