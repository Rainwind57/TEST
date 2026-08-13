"""模拟盘路由：下单、持仓查询、净值曲线、重置。"""
import datetime
from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel

from .. import adapters, db
from ..auth import get_user_id_from_auth
from .auth import require_user_id

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

COMMISSION = 0.00025   # 万 2.5 佣金（单边）
STAMP_DUTY = 0.001     # 千 1 印花税（卖出单边）


class OrderBody(BaseModel):
    code: str
    side: str  # "buy" | "sell" | "short" | "cover"
    qty: int


async def _build_portfolio_view():
    conn = db.get_conn()
    cash = conn.execute("SELECT cash FROM portfolio_state WHERE id = 1").fetchone()["cash"]
    positions = conn.execute("SELECT code, name, qty, avg_cost, side FROM positions").fetchall()
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
        side = p["side"] or "long"
        # 空头为负债：市值取负，盈亏 =（开仓价 - 现价）* 数量
        sign = -1.0 if side == "short" else 1.0
        value = sign * price * p["qty"]
        market_value += value
        pnl = (p["avg_cost"] - price) * p["qty"] if side == "short" else (price - p["avg_cost"]) * p["qty"]
        pnl_pct = ((p["avg_cost"] / price - 1) * 100 if side == "short" else (price / p["avg_cost"] - 1) * 100) if p["avg_cost"] else 0.0
        pos_list.append({
            "code": p["code"], "name": p["name"], "qty": p["qty"], "side": side,
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
async def place_order(body: OrderBody, uid: int = Depends(require_user_id)):
    if body.side not in ("buy", "sell", "short", "cover"):
        raise HTTPException(400, "side 必须为 buy/sell/short/cover")
    if body.qty <= 0 or body.qty % 100 != 0:
        raise HTTPException(400, "数量需为 100 的整数倍")

    code = body.code.strip().lower()
    if not db.is_tradable(code):
        raise HTTPException(400, f"无法交易非交易标的（{code} 为指数/ETF），仅支持可交易个股")
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
    pos_side = pos["side"] if pos else "long"

    if body.side == "buy":
        if pos and pos_side == "short":
            conn.close()
            raise HTTPException(422, "该代码持有空单，请先回补（side=cover）")
        cost = amount * COMMISSION
        if amount + cost > cash:
            conn.close()
            raise HTTPException(422, f"资金不足（含佣金{COMMISSION:.2%}）")
        new_cash = cash - amount - cost
        if pos:
            new_qty = pos["qty"] + body.qty
            new_avg_cost = (pos["avg_cost"] * pos["qty"] + amount + cost) / new_qty
            cur.execute("UPDATE positions SET qty = ?, avg_cost = ?, name = ? WHERE code = ?",
                        (new_qty, new_avg_cost, q["name"], code))
        else:
            cur.execute("INSERT INTO positions (code, name, qty, avg_cost, side) VALUES (?, ?, ?, ?, 'long')",
                        (code, q["name"], body.qty, price + cost / body.qty))
    elif body.side == "sell":
        if not pos or pos_side != "long" or pos["qty"] < body.qty:
            conn.close()
            raise HTTPException(422, "多头持仓不足")
        cost = amount * (COMMISSION + STAMP_DUTY)
        new_cash = cash + amount - cost
        remain = pos["qty"] - body.qty
        if remain <= 0:
            cur.execute("DELETE FROM positions WHERE code = ?", (code,))
        else:
            cur.execute("UPDATE positions SET qty = ? WHERE code = ?", (remain, code))
    elif body.side == "short":
        # 融券开空：按现价"卖出借入股份"，现金增加，负债以负市值体现
        if pos and pos_side == "long":
            conn.close()
            raise HTTPException(422, "该代码持有多头，请先卖出（side=sell）后再做空")
        cost = amount * COMMISSION
        new_cash = cash + amount - cost
        if pos:
            new_qty = pos["qty"] + body.qty
            new_avg_cost = (pos["avg_cost"] * pos["qty"] + amount) / new_qty
            cur.execute("UPDATE positions SET qty = ?, avg_cost = ?, name = ?, side = 'short' WHERE code = ?",
                        (new_qty, new_avg_cost, q["name"], code))
        else:
            cur.execute("INSERT INTO positions (code, name, qty, avg_cost, side) VALUES (?, ?, ?, ?, 'short')",
                        (code, q["name"], body.qty, price))
    else:  # cover：买入回补空单
        if not pos or pos_side != "short" or pos["qty"] < body.qty:
            conn.close()
            raise HTTPException(422, "空头持仓不足")
        cost = amount * COMMISSION
        new_cash = cash - amount - cost
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
