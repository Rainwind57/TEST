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

    betas, factor_names, X, codes_valid = risk.build_style_panel(stock_data)
    if X.size == 0 or betas.size == 0:
        raise HTTPException(422, "有效持仓样本不足，无法构建风格面板（需≥3只且历史≥30日）")

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

    # Fama-MacBeth：因子收益取各截面回归 beta 的时序均值，协方差用 beta 时序估计
    # （旧版用截面暴露 np.cov(X) 冒充因子收益协方差，对象错误）
    factor_returns = betas.mean(axis=0)
    sigma_f = risk.factor_covariance(betas)
    # 逐股特质方差（旧版用截面残差方差标量广播到所有股票，过粗）
    sigma_e = risk.estimate_specific_variances(stock_data, X, codes_valid, factor_returns)
    if sigma_e.size == 0 or np.all(sigma_e == 0):
        # 退化兜底：截面残差方差
        residuals = stock_returns - X @ factor_returns
        sigma_e = np.var(residuals, ddof=max(1, len(residuals) - X.shape[1] - 1)) * np.ones(len(codes_valid))

    attr = risk.attribute_returns(weights, X, factor_returns, stock_returns)
    rdecomp = risk.risk_decomposition(weights, X, sigma_f, sigma_e)

    # VaR/CVaR（历史模拟法）：用个股 K 线收益时序 + 当前权重算组合收益分布
    var_result = _compute_portfolio_var(stock_data, codes_valid, weights)
    sample_warning = (None if len(codes_valid) >= 30
                      else f"持仓仅 {len(codes_valid)} 只（<30），风格回归自由度不足，结果仅供参考")

    return {
        "holdings": [{"code": c, "weight": float(w)} for c, w in zip(codes_valid, weights)],
        "factorNames": factor_names,
        "exposures": attr["exposures"],
        "factorContribution": attr["factorContribution"],
        "totalReturn": attr["total"],
        "factorReturn": attr["totalFactor"],
        "residual": attr["residual"],
        "risk": rdecomp,
        "var": var_result,
        "sampleWarning": sample_warning,
    }


def _compute_portfolio_var(stock_data: list[dict], codes_valid: list[str],
                           weights: np.ndarray) -> dict:
    """用各股 K 线收益时序 + 当前权重算组合收益分布，给历史模拟 VaR/CVaR。

    对齐各股收益长度到最短，缺失股跳过；样本不足返回告警而非崩。
    """
    by_code = {s["code"]: s for s in stock_data}
    series_list = []
    for c in codes_valid:
        s = by_code.get(c, {})
        kline = s.get("kline", [])
        if len(kline) < 30:
            continue
        try:
            arr = risk.kline_to_arrays(kline)
        except Exception:
            continue
        closes = arr["close"]
        rets = np.diff(closes) / closes[:-1]
        rets = rets[np.isfinite(rets)]
        series_list.append((c, rets))
    if len(series_list) < 3:
        return {"var": 0.0, "cvar": 0.0, "n": 0, "warning": "有效个股收益序列不足，需≥3只且历史≥30日"}
    min_len = min(len(r) for _, r in series_list)
    mat = np.array([r[-min_len:] for _, r in series_list], dtype=np.float64).T
    w = np.array([weights[codes_valid.index(c)] for c, _ in series_list], dtype=np.float64)
    w = w / w.sum() if w.sum() > 0 else w
    return risk.value_at_risk(mat, w, alpha=0.05)
