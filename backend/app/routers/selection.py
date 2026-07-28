"""全市场选股：多因子加权打分选股 + 因子分层回测 + 一键落地到自选/模拟盘。"""
import asyncio
import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import adapters, db
from ..factors import (
    FACTORS, SNAPSHOT_FACTORS, snapshot_factor_value, composite_score,
    bucket_index, pearson, spearman, mean,
)
from .portfolio import OrderBody, place_order

router = APIRouter(prefix="/api/select", tags=["select"])

BOARD_LABELS = {
    "all": "全部A股", "sh_main": "沪市主板", "sz_main": "深市主板",
    "gem": "创业板", "star": "科创板", "bse": "北交所",
}


@router.get("/boards")
def list_boards():
    return [{"value": k, "label": v} for k, v in BOARD_LABELS.items()]


@router.get("/factors")
def list_select_factors():
    items = []
    for key, meta in SNAPSHOT_FACTORS.items():
        items.append({"key": key, "label": meta["label"], "group": meta["group"], "direction": meta["direction"], "format": meta["format"], "kline": False})
    for key, meta in FACTORS.items():
        items.append({"key": key, "label": meta["label"], "group": meta["group"], "direction": meta["direction"], "format": meta["format"], "kline": True})
    return items


@router.get("/market")
async def market_list(board: str = "all", limit: int = 100, sortField: str = "f6"):
    try:
        rows = await adapters.fetch_market_list(board, limit, sortField)
    except Exception as e:
        raise HTTPException(502, f"行情列表获取失败: {e}")
    return rows


class FactorSpec(BaseModel):
    key: str
    weight: float = 1.0
    direction: int = 1


class FilterSpec(BaseModel):
    excludeSt: bool = True
    minPrice: float | None = None
    maxPrice: float | None = None
    minPe: float | None = None
    maxPe: float | None = None
    minMktCap: float | None = None
    maxMktCap: float | None = None


class SelectBody(BaseModel):
    board: str = "all"
    poolSize: int = 200
    factors: list[FactorSpec]
    topN: int = 20
    filters: FilterSpec = FilterSpec()


async def _fetch_technical_values(codes: list[str], keys: list[str], hist: int = 260) -> dict:
    sem = asyncio.Semaphore(15)

    async def one(code):
        async with sem:
            try:
                kline = await adapters.fetch_kline(code, hist)
            except Exception:
                return code, {}
            if len(kline) < 25:
                return code, {}
            i = len(kline) - 1
            vals = {key: FACTORS[key]["calc"](kline, i) for key in keys if key in FACTORS}
            return code, vals

    results = await asyncio.gather(*(one(c) for c in codes))
    return dict(results)


@router.post("")
async def run_select(body: SelectBody):
    if not body.factors:
        raise HTTPException(400, "请至少选择一个因子")
    unknown = [s.key for s in body.factors if s.key not in FACTORS and s.key not in SNAPSHOT_FACTORS]
    if unknown:
        raise HTTPException(400, f"未知因子: {unknown}")

    try:
        pool = await adapters.fetch_market_list(body.board, body.poolSize)
    except Exception as e:
        raise HTTPException(502, f"候选池获取失败: {e}")

    f = body.filters
    candidates = []
    for row in pool:
        if f.excludeSt and "ST" in (row["name"] or "").upper():
            continue
        if f.minPrice is not None and (row["price"] or 0) < f.minPrice:
            continue
        if f.maxPrice is not None and (row["price"] or 0) > f.maxPrice:
            continue
        if f.minPe is not None and (row["pe"] is None or row["pe"] < f.minPe):
            continue
        if f.maxPe is not None and (row["pe"] is None or row["pe"] > f.maxPe):
            continue
        if f.minMktCap is not None and (row["mktCap"] or 0) < f.minMktCap:
            continue
        if f.maxMktCap is not None and (row["mktCap"] or 0) > f.maxMktCap:
            continue
        candidates.append(row)

    if not candidates:
        raise HTTPException(422, "过滤后候选池为空，请放宽筛选条件")

    tech_keys = [s.key for s in body.factors if s.key in FACTORS]
    snap_keys = [s.key for s in body.factors if s.key in SNAPSHOT_FACTORS]

    tech_values = {}
    if tech_keys:
        tech_values = await _fetch_technical_values([c["code"] for c in candidates], tech_keys)

    scored_rows = []
    for row in candidates:
        entry = dict(row)
        for key in snap_keys:
            entry[key] = snapshot_factor_value(row, key)
        for key in tech_keys:
            entry[key] = tech_values.get(row["code"], {}).get(key)
        scored_rows.append(entry)

    specs = [{"key": s.key, "weight": s.weight, "direction": s.direction} for s in body.factors]
    scored = composite_score(scored_rows, specs)
    scored.sort(key=lambda r: r["score"], reverse=True)
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank

    return {
        "universeSize": len(pool), "candidateSize": len(candidates),
        "rows": scored[: max(1, body.topN)],
    }


class BacktestBody(BaseModel):
    board: str = "all"
    poolSize: int = 60
    factor: str = "momentum"
    groups: int = 5
    n: int = 5
    hist: int = 180


@router.post("/backtest")
async def run_backtest(body: BacktestBody):
    if body.factor not in FACTORS:
        raise HTTPException(400, "分层回测目前仅支持技术类(量价)因子")
    groups = max(2, min(10, body.groups))
    n = max(1, body.n)
    hist = max(60, body.hist)

    try:
        pool = await adapters.fetch_market_list(body.board, body.poolSize)
    except Exception as e:
        raise HTTPException(502, f"候选池获取失败: {e}")

    codes = [row["code"] for row in pool]
    sem = asyncio.Semaphore(15)

    async def fetch_one(code):
        async with sem:
            try:
                kline = await adapters.fetch_kline(code, hist)
            except Exception:
                return code, []
            return code, kline

    fetched = await asyncio.gather(*(fetch_one(c) for c in codes))
    series = {code: kl for code, kl in fetched if len(kl) >= 40}
    if len(series) < groups * 3:
        raise HTTPException(422, "有效股票样本不足，请增大候选池规模")

    calc = FACTORS[body.factor]["calc"]
    date_maps = {code: {row["date"]: idx for idx, row in enumerate(kl)} for code, kl in series.items()}
    ref_code = max(series, key=lambda c: len(series[c]))
    ref_dates = [row["date"] for row in series[ref_code]]

    ic_series = []
    bucket_returns = [[] for _ in range(groups)]
    long_short_points = []
    cum = 1.0

    for t in range(25, len(ref_dates) - n, max(1, n)):
        date_t = ref_dates[t]
        cross = []
        for code, kl in series.items():
            i = date_maps[code].get(date_t)
            if i is None or i < 20 or i + n >= len(kl):
                continue
            closes = [row["close"] for row in kl]
            fv = calc(kl, i)
            if fv is None or closes[i] == 0:
                continue
            fret = closes[i + n] / closes[i] - 1
            cross.append((code, fv, fret))
        if len(cross) < groups * 2:
            continue

        fvs = [c[1] for c in cross]
        rets = [c[2] for c in cross]
        ic = pearson(fvs, rets)
        rank_ic = spearman(fvs, rets)
        ic_series.append({"date": date_t, "ic": ic, "rankIc": rank_ic, "sample": len(cross)})

        sorted_fvs = sorted(fvs)
        day_buckets = [[] for _ in range(groups)]
        for _, fv, fret in cross:
            b = bucket_index(fv, sorted_fvs, groups)
            day_buckets[b].append(fret)
            bucket_returns[b].append(fret)

        if day_buckets[0] and day_buckets[-1]:
            top_ret = mean(day_buckets[-1])
            bottom_ret = mean(day_buckets[0])
            cum *= (1 + top_ret - bottom_ret)
            long_short_points.append({"date": date_t, "longShort": top_ret - bottom_ret, "cum": cum - 1})

    if not ic_series:
        raise HTTPException(422, "有效截面样本不足，无法完成分层回测")

    group_summary = [
        {"group": idx + 1, "avgReturn": mean(rets) if rets else 0.0, "sample": len(rets)}
        for idx, rets in enumerate(bucket_returns)
    ]
    mean_ic = mean([p["ic"] for p in ic_series])
    mean_rank_ic = mean([p["rankIc"] for p in ic_series])
    ic_win_rate = sum(1 for p in ic_series if p["ic"] > 0) / len(ic_series)

    return {
        "factorLabel": FACTORS[body.factor]["label"], "groups": groups, "n": n,
        "meanIc": mean_ic, "meanRankIc": mean_rank_ic, "icWinRate": ic_win_rate,
        "icSeries": ic_series, "groupSummary": group_summary, "longShort": long_short_points,
        "universeSize": len(pool), "effectiveStocks": len(series), "rebalanceCount": len(ic_series),
    }


class ApplyBody(BaseModel):
    codes: list[str]
    action: str  # "watchlist" | "buy"
    totalCash: float | None = None


@router.post("/apply")
async def apply_selection(body: ApplyBody):
    if not body.codes:
        raise HTTPException(400, "codes 不能为空")

    if body.action == "watchlist":
        conn = db.get_conn()
        added = 0
        for code in body.codes:
            code = code.strip().lower()
            existing = conn.execute("SELECT 1 FROM watchlist WHERE code = ?", (code,)).fetchone()
            if existing:
                continue
            conn.execute("INSERT INTO watchlist (code, added_at) VALUES (?, ?)",
                         (code, datetime.datetime.now().isoformat()))
            added += 1
        conn.commit()
        conn.close()
        return {"ok": True, "added": added}

    if body.action == "buy":
        if not body.totalCash or body.totalCash <= 0:
            raise HTTPException(400, "请提供买入总资金 totalCash")
        per_code_cash = body.totalCash / len(body.codes)
        codes = [c.strip().lower() for c in body.codes]
        quotes = await adapters.fetch_tencent_quotes(codes)
        results = []
        for code in codes:
            q = quotes.get(code)
            if not q or not q.get("price"):
                results.append({"code": code, "ok": False, "reason": "无行情"})
                continue
            qty = int(per_code_cash / q["price"] // 100) * 100
            if qty <= 0:
                results.append({"code": code, "ok": False, "reason": "资金不足一手"})
                continue
            try:
                await place_order(OrderBody(code=code, side="buy", qty=qty))
                results.append({"code": code, "ok": True, "qty": qty})
            except HTTPException as e:
                results.append({"code": code, "ok": False, "reason": e.detail})
        return {"ok": True, "results": results}

    raise HTTPException(400, "action 必须为 watchlist 或 buy")
