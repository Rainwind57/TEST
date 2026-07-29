"""风险归因路由：对当前持仓做 Barra 风格风险分解。"""
import asyncio
from fastapi import APIRouter, HTTPException
import numpy as np

from .. import adapters, db, risk

router = APIRouter(prefix="/api/risk", tags=["risk"])


@router.get("/attribution")
async def attribution():
    """对当前模拟盘持仓做收益归因 + 风险分解。"""
    conn = db.get_conn()
    positions = conn.execute("SELECT code, name, qty, avg_cost FROM positions").fetchall()
    state = conn.execute("SELECT cash FROM portfolio_state WHERE id = 1").fetchone()
    conn.close()
    if not positions:
        raise HTTPException(422, "当前无持仓，无法做风险归因")

    codes = [r["code"] for r in positions]
    quotes = await adapters.fetch_quotes(codes)
    stock_data = []
    for p in positions:
        q = quotes.get(p["code"], {})
        try:
            kline = await adapters.fetch_kline(p["code"], 60)
        except Exception:
            kline = []
        stock_data.append({"code": p["code"], "quote": q, "kline": kline})

    X, codes_valid, factor_names = risk.build_style_matrix(stock_data)
    if X.size == 0:
        raise HTTPException(422, "有效持仓样本不足，无法构建风格矩阵")

    prices = np.array([quotes.get(c, {}).get("price", 0) for c in codes_valid])
    qty = np.array([next(p["qty"] for p in positions if p["code"] == c) for c in codes_valid])
    mv = prices * qty
    if mv.sum() == 0:
        raise HTTPException(422, "持仓市值为 0")
    weights = mv / mv.sum()
    stock_returns = np.array([
        (quotes.get(c, {}).get("price", 0) / (next(p["avg_cost"] for p in positions if p["code"] == c)) - 1)
        if next(p["avg_cost"] for p in positions if p["code"] == c) else 0.0
        for c in codes_valid
    ])

    factor_returns = risk.cross_section_regression(X, stock_returns)
    sigma_f = risk.factor_covariance(X)
    residuals = stock_returns - X @ factor_returns
    sigma_e = np.var(residuals, ddof=max(1, len(residuals) - X.shape[1] - 1)) * np.ones(len(codes_valid))

    attr = risk.attribute_returns(weights, X, factor_returns, stock_returns)
    rdecomp = risk.risk_decomposition(weights, X, sigma_f, sigma_e)

    return {
        "holdings": [{"code": c, "weight": float(w)} for c, w in zip(codes_valid, weights)],
        "factorNames": factor_names,
        "exposures": attr["exposures"],
        "factorContribution": attr["factorContribution"],
        "totalReturn": attr["total"],
        "factorReturn": attr["totalFactor"],
        "residual": attr["residual"],
        "risk": rdecomp,
    }
