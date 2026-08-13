"""风险归因路由：对持仓组合做 Barra 风格风险分解。

联通设计：默认对当前模拟盘持仓归因；POST 可传任意 codes/weights 组合
（如选股结果、ML 打分 topN、组合优化输出），打通"选股→组合→风险"链路。
"""
import asyncio
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import numpy as np

from .auth import require_user_id

from .. import adapters, db, risk

router = APIRouter(prefix="/api/risk", tags=["risk"])


class AttributionBody(BaseModel):
    codes: list[str]              # 组合标的（如选股结果/ML打分/组合优化输出的 codes）
    weights: list[float] | None = None   # 可选：等权时忽略
    name: str = "自定义组合"


async def _run_attribution(codes: list[str], weights: list[float] | None = None):
    if not codes:
        raise HTTPException(422, "codes 不能为空")
    quotes = await adapters.fetch_quotes(codes)
    stock_data = []
    sem = asyncio.Semaphore(min(50, max(15, len(codes))))

    async def one(code):
        async with sem:
            try:
                kline = await adapters.fetch_kline(code, 60)
            except Exception:
                kline = []
            return {"code": code, "quote": quotes.get(code, {}), "kline": kline}
    stock_data = await asyncio.gather(*(one(c) for c in codes))

    betas, factor_names, X, codes_valid = await risk.build_style_panel(stock_data)
    if X.size == 0 or betas.size == 0:
        raise HTTPException(422, "有效组合样本不足，无法构建风格面板（需≥3只且历史≥30日）")

    prices = np.array([quotes.get(c, {}).get("price", 0) for c in codes_valid])
    if weights is not None:
        w_map = {c: float(wt) for c, wt in zip(codes, weights) if c in set(codes_valid)}
        w = np.array([w_map.get(c, 0.0) for c in codes_valid])
        dropped = len(codes) - sum(1 for c in codes if c in set(codes_valid))
        if w.sum() <= 0:
            w = np.ones(len(codes_valid))
    else:
        w = np.ones(len(codes_valid))
        dropped = 0
    mv = prices
    valid_mv = mv > 0
    if not valid_mv.any():
        raise HTTPException(422, "组合标的均无行情价格")
    w = w * valid_mv
    if w.sum() <= 0:
        raise HTTPException(422, "组合权重计算失败（无有效价格）")
    weights_w = w / w.sum()
    stock_returns = np.array([
        (quotes.get(c, {}).get("price", 0) / (quotes.get(c, {}).get("preClose", 0) or 1) - 1)
        if quotes.get(c, {}).get("preClose") else 0.0
        for c in codes_valid
    ])

    factor_returns = betas.mean(axis=0)
    sigma_f = risk.factor_covariance(betas)
    sigma_e = risk.estimate_specific_variances(stock_data, X, codes_valid, factor_returns)
    if sigma_e.size == 0 or np.all(sigma_e == 0):
        residuals = stock_returns - X @ factor_returns
        # ddof 不能超过样本数，否则 np.var 返回 NaN；行业因子多时 ddof 可能偏大
        safe_ddof = min(X.shape[1] + 1, max(1, len(residuals) - 1))
        sigma_e_scalar = np.var(residuals, ddof=safe_ddof)
        sigma_e = np.full(len(codes_valid), max(sigma_e_scalar, 1e-12))

    attr = risk.attribute_returns(weights_w, X, factor_returns, stock_returns)
    rdecomp = risk.risk_decomposition(weights_w, X, sigma_f, sigma_e)
    var_result = _compute_portfolio_var(stock_data, codes_valid, weights_w)
    sample_warning = (None if len(codes_valid) >= 30
                      else f"组合仅 {len(codes_valid)} 只（<30），风格回归自由度不足，结果仅供参考")

    snapshot_warning = (
        "风格因子中 turnover/ep/bp/size 使用当前最新值（快照），"
        "应用到所有历史截面时存在前视偏差（look-ahead bias）。"
        "因子暴露与风险分解仅供截面比较，不反映历史时序真实变化。"
    )

    dropped_codes = [c for c in codes if c not in set(codes_valid)] if weights is not None else []
    return {
        "holdings": [{"code": c, "weight": float(wt)} for c, wt in zip(codes_valid, weights_w)],
        "droppedCodes": dropped_codes,
        "snapshotWarning": snapshot_warning,
        "factorNames": factor_names,
        "factorLabels": risk.all_factor_labels(),
        "exposures": attr["exposures"],
        "factorContribution": attr["factorContribution"],
        "totalReturn": attr["total"],
        "factorReturn": attr["totalFactor"],
        "residual": attr["residual"],
        "risk": rdecomp,
        "var": var_result,
        "sampleWarning": sample_warning,
    }


@router.get("/attribution")
async def attribution():
    """对当前模拟盘持仓做收益归因 + 风险分解。"""
    conn = db.get_conn()
    positions = conn.execute("SELECT code, name, qty, avg_cost, side FROM positions").fetchall()
    conn.close()
    if not positions:
        raise HTTPException(422, "当前无持仓，无法做风险归因")
    codes = [r["code"] for r in positions]
    try:
        quotes = await adapters.fetch_tencent_quotes(codes)
    except Exception:
        quotes = {}
    mv = {r["code"]: ((quotes.get(r["code"], {}).get("price", 0) or r["avg_cost"]) * r["qty"]
                      * (-1.0 if r["side"] == "short" else 1.0))
          for r in positions}
    weights = [float(mv[c]) for c in codes]
    return await _run_attribution(codes, weights)


@router.post("/attribution")
async def attribution_custom(body: AttributionBody, uid: int = Depends(require_user_id)):
    """对传入的任意组合做风险归因（打通：选股结果/ML打分/组合优化 → 风险）。"""
    return await _run_attribution(body.codes, body.weights)


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
