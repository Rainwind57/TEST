"""组合优化路由：均值-方差/最大 Sharpe/风险平价，带个股权重上限约束。

修复：
- OptBody 加 codes 字段（旧版只有 mu/cov 向量，权重无法对应股票、无法落地模拟盘）
- mu/cov NaN/Inf 校验（旧版无校验，NaN 导致 cvxpy 抛 500）
- 新增 /apply：按优化权重一键建仓模拟盘（旧版权重结果只能画饼图，被丢弃）
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import asyncio
import numpy as np

from .. import portfolio_opt as po, db, adapters
from .portfolio import OrderBody, place_order
from .auth import require_user_id

router = APIRouter(prefix="/api/portfolio-opt", tags=["portfolio-opt"])


class OptBody(BaseModel):
    codes: list[str] = []         # 对应股票代码（与 mu 同长，用于落地模拟盘）
    mu: list[float]
    cov: list[list[float]]
    method: str = "mean_variance"  # mean_variance | max_sharpe | risk_parity | equal
    maxWeight: float = 0.1
    longOnly: bool = True
    rf: float = 0.0
    targetReturn: float | None = None


@router.post("")
def optimize(body: OptBody, uid: int = Depends(require_user_id)):
    mu = np.array(body.mu, dtype=np.float64)
    cov = np.array(body.cov, dtype=np.float64)
    if mu.ndim != 1 or cov.ndim != 2 or mu.shape[0] != cov.shape[0] != cov.shape[1]:
        raise HTTPException(400, "mu 与 cov 维度不匹配")
    if np.isnan(mu).any() or np.isnan(cov).any() or np.isinf(mu).any() or np.isinf(cov).any():
        raise HTTPException(400, "mu/cov 含 NaN 或 Inf，请检查输入")
    if body.codes and len(body.codes) != len(mu):
        raise HTTPException(400, "codes 长度需与 mu 一致")

    method = body.method
    if method == "mean_variance":
        w = po.mean_variance(mu, cov, body.maxWeight, body.longOnly, body.targetReturn)
    elif method == "max_sharpe":
        w = po.max_sharpe(mu, cov, body.maxWeight, body.rf, body.longOnly)
    elif method == "risk_parity":
        w = po.risk_parity(cov, body.maxWeight)
    elif method == "equal":
        w = po.equal_weight(len(mu))
    else:
        raise HTTPException(400, f"未知方法: {method}")

    stats = po.portfolio_stats(w, mu, cov, body.rf)
    return {"codes": body.codes, "weights": w.tolist(), "stats": stats, "method": method}


class ApplyBody(BaseModel):
    codes: list[str]
    weights: list[float]
    totalAssets: float | None = None  # 可选；默认用模拟盘总资产


@router.post("/apply")
async def apply_to_portfolio(body: ApplyBody, uid: int = Depends(require_user_id)):
    """按优化权重建仓模拟盘（多头权重买入，负权重做空）。"""
    if len(body.codes) != len(body.weights):
        raise HTTPException(400, "codes 与 weights 长度不一致")
    w = np.array(body.weights, dtype=np.float64)
    if len(w) == 0:
        raise HTTPException(400, "codes 不能为空")
    # 多头/空头各自归一化到 1：多空对冲组合 w.sum() 可能 ≈0，不能整体归一
    long_w = np.clip(w, 0, None)
    short_w = np.clip(-w, 0, None)
    long_sum = float(long_w.sum())
    short_sum = float(short_w.sum())
    if long_sum <= 0 and short_sum <= 0:
        raise HTTPException(400, "权重需至少一侧为正")
    if long_sum > 0:
        long_w = long_w / long_sum
    if short_sum > 0:
        short_w = short_w / short_sum
    long_target = {code: float(wi) for code, wi in zip(body.codes, long_w) if wi > 0}
    short_target = {code: float(wi) for code, wi in zip(body.codes, short_w) if wi > 0}

    with db.get_conn() as conn:
        cash = conn.execute("SELECT cash FROM portfolio_state WHERE id = 1").fetchone()["cash"]
        positions_rows = conn.execute("SELECT code, qty, side FROM positions").fetchall()
    long_pos = {p["code"]: p["qty"] for p in positions_rows if (p["side"] or "long") == "long"}
    short_pos = {p["code"]: p["qty"] for p in positions_rows if p["side"] == "short"}

    all_codes = list(set(body.codes) | set(long_pos.keys()) | set(short_pos.keys()))
    quotes = await adapters.fetch_tencent_quotes(all_codes)
    long_mv = sum(quotes.get(c, {}).get("price", 0) * q for c, q in long_pos.items())
    short_mv = sum(quotes.get(c, {}).get("price", 0) * q for c, q in short_pos.items())
    total = body.totalAssets if body.totalAssets else (cash + long_mv)

    applied = []
    # 先释放不在目标内的持仓：多头卖、空头回补
    for code, qty in long_pos.items():
        if code not in long_target:
            try:
                await place_order(OrderBody(code=code, side="sell", qty=qty))
                applied.append({"code": code, "side": "sell", "qty": qty, "reason": "调出多头组合"})
            except Exception as e:
                applied.append({"code": code, "side": "sell", "error": str(e)})
    for code, qty in short_pos.items():
        if code not in short_target:
            try:
                await place_order(OrderBody(code=code, side="cover", qty=qty))
                applied.append({"code": code, "side": "cover", "qty": qty, "reason": "调出空头组合"})
            except Exception as e:
                applied.append({"code": code, "side": "cover", "error": str(e)})

    # 多头腿：买/卖调整
    for code, wi in long_target.items():
        price = quotes.get(code, {}).get("price", 0)
        if price <= 0:
            applied.append({"code": code, "weight": wi, "error": "无行情"})
            continue
        target_qty = int(total * wi / price / 100) * 100
        current_qty = long_pos.get(code, 0)
        if target_qty < current_qty:
            sell_qty = (current_qty - target_qty) // 100 * 100
            if sell_qty >= 100:
                try:
                    await place_order(OrderBody(code=code, side="sell", qty=sell_qty))
                    applied.append({"code": code, "side": "sell", "qty": sell_qty, "weight": wi})
                except Exception as e:
                    applied.append({"code": code, "weight": wi, "error": str(e)})
        elif target_qty > current_qty:
            buy_qty = (target_qty - current_qty) // 100 * 100
            if buy_qty < 100:
                applied.append({"code": code, "weight": wi, "error": "调整量不足 1 手"})
                continue
            try:
                await place_order(OrderBody(code=code, side="buy", qty=buy_qty))
                applied.append({"code": code, "side": "buy", "qty": buy_qty, "weight": wi})
            except Exception as e:
                applied.append({"code": code, "weight": wi, "error": str(e)})
        else:
            applied.append({"code": code, "weight": wi, "action": "hold"})

    # 空头腿：short/cover 调整（负权重 → 做空）
    for code, wi in short_target.items():
        price = quotes.get(code, {}).get("price", 0)
        if price <= 0:
            applied.append({"code": code, "weight": -wi, "error": "无行情"})
            continue
        target_qty = int(total * wi / price / 100) * 100
        current_qty = short_pos.get(code, 0)
        if target_qty < current_qty:
            cover_qty = (current_qty - target_qty) // 100 * 100
            if cover_qty >= 100:
                try:
                    await place_order(OrderBody(code=code, side="cover", qty=cover_qty))
                    applied.append({"code": code, "side": "cover", "qty": cover_qty, "weight": -wi})
                except Exception as e:
                    applied.append({"code": code, "weight": -wi, "error": str(e)})
        elif target_qty > current_qty:
            short_qty = (target_qty - current_qty) // 100 * 100
            if short_qty < 100:
                applied.append({"code": code, "weight": -wi, "error": "调整量不足 1 手"})
                continue
            try:
                await place_order(OrderBody(code=code, side="short", qty=short_qty))
                applied.append({"code": code, "side": "short", "qty": short_qty, "weight": -wi})
            except Exception as e:
                applied.append({"code": code, "weight": -wi, "error": str(e)})
        else:
            applied.append({"code": code, "weight": -wi, "action": "hold"})
    return {"applied": applied, "totalAssets": total}


class EstimateBody(BaseModel):
    codes: list[str]
    hist: int = 120          # 历史长度（交易日）


@router.post("/estimate")
async def estimate_mu_cov(body: EstimateBody, uid: int = Depends(require_user_id)):
    """从历史 K 线自动估计 mu（年化预期收益）+ cov（年化协方差），
    取消用户手工粘贴矩阵。mu=日收益均值×252，cov=日收益协方差×252。"""
    if len(body.codes) < 2:
        raise HTTPException(400, "至少需要 2 只股票")
    sem = asyncio.Semaphore(min(50, max(15, len(body.codes))))

    async def fetch_one(code):
        async with sem:
            try:
                kline = await adapters.fetch_kline(code, body.hist + 5)
            except Exception:
                return code, []
            return code, kline

    fetched = await asyncio.gather(*(fetch_one(c) for c in body.codes))
    series = {c: kl for c, kl in fetched if len(kl) >= 30}
    if len(series) < 2:
        raise HTTPException(422, "有效股票历史不足，请增大 hist 或更换代码")

    codes_valid = [c for c in body.codes if c in series]
    # 按日期对齐而非按序列头部对齐：各股上市/停牌日期不一致时，头部截断会错位
    # 收集 (code → {date: close})，取所有股票公共交易日，按日期重排收益序列
    date_close = {c: {k["date"]: k["close"] for k in series[c] if k.get("date") and k.get("close")} for c in codes_valid}
    common_dates = sorted(set.intersection(*[set(d.keys()) for d in date_close.values()])) if date_close else []
    if len(common_dates) < 30:
        raise HTTPException(422, "公共交易日不足，各股历史区间差异过大")

    ret_matrix = []
    prev_close = {c: date_close[c][common_dates[0]] for c in codes_valid}
    for d in common_dates[1:]:
        row = []
        skip = False
        for c in codes_valid:
            pc = prev_close.get(c)
            cc = date_close[c].get(d)
            if not pc or not cc:
                skip = True
                break
            row.append(cc / pc - 1)
            prev_close[c] = cc
        if skip:
            continue
        ret_matrix.append(row)
    if len(ret_matrix) < 20:
        raise HTTPException(422, "有效收益样本不足")

    R = np.array(ret_matrix, dtype=np.float64)
    mu = R.mean(axis=0) * 252
    cov = np.cov(R, rowvar=False) * 252
    return {"codes": codes_valid, "mu": mu.tolist(), "cov": cov.tolist(), "alignedDays": len(ret_matrix)}
