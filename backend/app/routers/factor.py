"""因子截面与单因子回归路由。"""
import asyncio
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from .. import adapters
from ..factors import (
    FACTORS, SNAPSHOT_FACTORS, snapshot_factor_value, pearson, spearman,
    REGRESSION_METHODS, fit_regression, poly_predict,
)
from .auth import require_user_id

router = APIRouter(prefix="/api", tags=["factor"])


@router.get("/factors")
async def get_factors(codes: str):
    code_list = [c.strip() for c in codes.split(",") if c.strip()]
    if not code_list:
        raise HTTPException(400, "codes 不能为空")

    try:
        quotes = await adapters.fetch_tencent_quotes(code_list)
    except Exception:
        quotes = {}

    # 行业映射（一次性拉取）
    try:
        sector_map = await adapters.fetch_sector_map()
    except Exception:
        sector_map = {}

    # 财务/资金流/北向：行情快照不含，逐股并发拉取（旧版未拉 → 这些因子恒为 None）
    extra = {}
    sem = asyncio.Semaphore(10)

    async def fetch_extra(code):
        row = {}
        async with sem:
            try:
                fin = await adapters.fetch_finance_summary(code)
            except Exception:
                fin = {}
            row["roe"] = fin.get("roe")
            row["net_margin"] = fin.get("netMargin")
            row["revenue_yoy"] = fin.get("revenueYoY")
            row["profit_yoy"] = fin.get("profitYoY")
            row["gross_margin"] = fin.get("grossMargin")
            row["debt_ratio"] = fin.get("debtRatio")
            row["eps"] = fin.get("eps")
            row["bps"] = fin.get("bps")
            row["roa"] = fin.get("roa")
            try:
                mf = await adapters.fetch_money_flow(code)
            except Exception:
                mf = {}
            row["main_net_pct"] = mf.get("mainNetPct")
            try:
                nh = await adapters.fetch_north_holding(code)
            except Exception:
                nh = {}
            row["north_holding_pct"] = nh.get("holdRatio")
        extra[code] = row

    await asyncio.gather(*(fetch_extra(c) for c in code_list))

    rows = []
    for code in code_list:
        q = quotes.get(code)
        try:
            kline = await adapters.fetch_kline(code, 260)
        except Exception:
            kline = []
        if not kline:
            rows.append({"code": code, "name": q["name"] if q else code, "error": True})
            continue

        i = len(kline) - 1
        row = {"code": code, "name": q["name"] if q else code}
        for key, meta in FACTORS.items():
            row[key] = meta["calc"](kline, i)

        ex = extra.get(code, {})
        snap_row = {}
        if q:
            pct_chg = (q["price"] / q["preClose"] - 1) * 100 if q.get("preClose") else None
            snap_row = {
                "pctChg": pct_chg, "turnover": q.get("turnover"), "amount": q.get("amount"),
                "pe": q.get("pe"), "pb": q.get("pb"),
                "mktCap": q.get("mktCap"), "circMktCap": q.get("circMktCap"),
                **ex, "sector": sector_map.get(code, ""),
            }
        for key in SNAPSHOT_FACTORS:
            row[key] = snapshot_factor_value(snap_row, key) if snap_row else None

        rows.append(row)
    return rows


@router.get("/factors/catalog")
def factor_catalog():
    items = []
    for key, meta in SNAPSHOT_FACTORS.items():
        items.append({"key": key, "label": meta["label"], "group": meta["group"], "direction": meta["direction"], "format": meta["format"], "kline": False})
    for key, meta in FACTORS.items():
        items.append({"key": key, "label": meta["label"], "group": meta["group"], "direction": meta["direction"], "format": meta["format"], "kline": True})
    return items


class RegressionBody(BaseModel):
    codes: list[str]
    factor: str = "ma_dev"
    method: str = "ols"
    n: int = 5
    hist: int = 150


@router.get("/regression/methods")
def list_regression_methods():
    return [{"key": k, "label": v["label"]} for k, v in REGRESSION_METHODS.items()]


@router.post("/regression")
async def run_regression(body: RegressionBody, uid: int = Depends(require_user_id)):
    if body.factor not in FACTORS:
        raise HTTPException(400, "未知因子类型")
    if body.method not in REGRESSION_METHODS:
        raise HTTPException(400, "未知回归方法")
    if not body.codes:
        raise HTTPException(400, "codes 不能为空")

    factor = FACTORS[body.factor]
    n = max(1, body.n)
    hist = max(60, body.hist)

    xs: list[float] = []
    ys: list[float] = []
    from ..numpy_factors import kline_to_arrays, compute_factor_series, series_at
    for code in body.codes:
        try:
            kline = await adapters.fetch_kline(code, hist)
        except Exception:
            continue
        if len(kline) < 40:
            continue
        closes = [k["close"] for k in kline]
        # 向量化预计算因子全序列（旧版逐 i 重算 factor["calc"]，O(hist²)）；
        # 序列不支持时回退标量 calc
        arr = kline_to_arrays(kline)
        series = compute_factor_series(body.factor, arr)
        for i in range(20, len(closes) - n):
            fv = series_at(series, i) if series is not None else factor["calc"](kline, i)
            if fv is None:
                continue
            base = closes[i]
            if base == 0:
                continue
            fret = closes[i + n] / base - 1
            xs.append(fv)
            ys.append(fret)

    if len(xs) < 10:
        raise HTTPException(422, "样本不足，请增加自选股数量或历史长度")

    reg = fit_regression(body.method, xs, ys)
    ic = pearson(xs, ys)
    rank_ic = spearman(xs, ys)
    min_x, max_x = min(xs), max(xs)
    steps = 40
    span = (max_x - min_x) or 1e-9
    line = [
        {"x": min_x + span * k / steps, "y": poly_predict(reg["coefs"], min_x + span * k / steps)}
        for k in range(steps + 1)
    ]
    samples = [{"x": xs[i], "y": ys[i]} for i in range(len(xs))]
    return {
        "factorLabel": factor["label"], "methodLabel": REGRESSION_METHODS[body.method]["label"], "n": n,
        "coefs": reg["coefs"], "r2": reg["r2"], "sampleSize": reg["n"],
        "ic": ic, "rankIc": rank_ic,
        "samples": samples, "line": line,
    }
