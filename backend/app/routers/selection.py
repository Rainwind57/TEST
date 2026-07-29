"""全市场选股：多因子加权打分选股 + 因子分层回测 + 一键落地到自选/模拟盘。"""
import asyncio
import datetime
import math
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import adapters, db
from ..factors import (
    FACTORS, SNAPSHOT_FACTORS, snapshot_factor_value, composite_score,
    bucket_index, pearson, spearman, mean, std, zscore, multi_ols,
    annualized_return, annualized_volatility, sharpe_ratio, sortino_ratio,
    max_drawdown, calmar_ratio, win_rate, information_coefficient_stats,
    round_trip_cost_rate, compute_user_factor_scores,
)
from .portfolio import OrderBody, place_order

router = APIRouter(prefix="/api/select", tags=["select"])

BOARD_LABELS = {
    "all": "全部A股", "sh_main": "沪市主板", "sz_main": "深市主板",
    "gem": "创业板", "star": "科创板", "bse": "北交所",
}

BENCHMARKS = {
    "none": None,
    "hs300": "sh000300",
    "zz500": "sh000905",
    "sse": "sh000001",
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
    builtin_keys = set(FACTORS) | set(SNAPSHOT_FACTORS)
    uf_specs = [s for s in body.factors if s.key.startswith("uf:")]
    unknown = [s.key for s in body.factors if s.key not in builtin_keys and not s.key.startswith("uf:")]
    if unknown:
        raise HTTPException(400, f"未知因子: {unknown}")

    uf_defs = {}
    for s in uf_specs:
        try:
            uf_id = int(s.key.split(":", 1)[1])
        except (ValueError, IndexError):
            raise HTTPException(400, f"无效的自定义因子标识: {s.key}")
        uf = db.get_user_factor(uf_id)
        if not uf:
            raise HTTPException(400, f"自定义因子不存在: {s.key}")
        uf_defs[s.key] = uf["definition"]

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

    for uf_key, definition in uf_defs.items():
        scores = compute_user_factor_scores(scored_rows, definition)
        for idx, row in enumerate(scored_rows):
            row[uf_key] = scores[idx]

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
    commissionRate: float = 0.00025   # 佣金费率（万 2.5，单边）
    stampDuty: float = 0.001          # 印花税（千 1，卖出单边）
    slippage: float = 0.001           # 滑点/冲击成本（单边）
    benchmark: str = "none"           # none | hs300 | zz500 | sse
    applyCost: bool = True


@router.post("/backtest")
async def run_backtest(body: BacktestBody):
    if body.factor not in FACTORS:
        raise HTTPException(400, "分层回测目前仅支持技术类(量价)因子")
    groups = max(2, min(10, body.groups))
    n = max(1, body.n)
    hist = max(60, body.hist)

    bench_code = BENCHMARKS.get(body.benchmark)
    if body.benchmark != "none" and bench_code is None:
        raise HTTPException(400, f"未知基准: {body.benchmark}")

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

    bench_series = None
    if bench_code:
        try:
            bench_series = await adapters.fetch_kline(bench_code, hist + n + 25)
        except Exception:
            bench_series = None

    calc = FACTORS[body.factor]["calc"]
    date_maps = {code: {row["date"]: idx for idx, row in enumerate(kl)} for code, kl in series.items()}
    ref_code = max(series, key=lambda c: len(series[c]))
    ref_dates = [row["date"] for row in series[ref_code]]

    cost_rate = round_trip_cost_rate(body.commissionRate, body.stampDuty, body.slippage) if body.applyCost else 0.0

    ic_series = []
    bucket_returns = [[] for _ in range(groups)]
    long_short_points = []
    top_group_returns = []
    bench_returns = []
    cum = 1.0
    cum_top = 1.0

    bench_date_idx = {}
    if bench_series:
        bench_date_idx = {row["date"]: idx for idx, row in enumerate(bench_series)}

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
            ls_ret = top_ret - bottom_ret
            net_ls = ls_ret - cost_rate
            net_top = top_ret - cost_rate
            cum *= (1.0 + net_ls)
            cum_top *= (1.0 + net_top)
            long_short_points.append({
                "date": date_t, "longShort": net_ls, "cum": cum - 1.0,
                "topCum": cum_top - 1.0, "gross": ls_ret,
            })
            top_group_returns.append(net_top)
            if bench_series and bench_date_idx:
                bi = bench_date_idx.get(date_t)
                if bi is not None and bi + n < len(bench_series):
                    bc = [row["close"] for row in bench_series]
                    bench_returns.append(bc[bi + n] / bc[bi] - 1)

    if not ic_series:
        raise HTTPException(422, "有效截面样本不足，无法完成分层回测")

    group_summary = [
        {"group": idx + 1, "avgReturn": mean(rets) if rets else 0.0, "sample": len(rets)}
        for idx, rets in enumerate(bucket_returns)
    ]

    ls_returns = [p["longShort"] for p in long_short_points]
    ls_equity = [1.0]
    for r in ls_returns:
        ls_equity.append(ls_equity[-1] * (1.0 + r))

    metrics = {
        "cumulativeReturn": cum - 1.0,
        "annualizedReturn": annualized_return(ls_returns),
        "annualizedVolatility": annualized_volatility(ls_returns),
        "sharpe": sharpe_ratio(ls_returns),
        "sortino": sortino_ratio(ls_returns),
        "maxDrawdown": max_drawdown(ls_equity),
        "calmar": calmar_ratio(ls_returns),
        "winRate": win_rate(ls_returns),
        "topGroupCumReturn": cum_top - 1.0,
        "topGroupAnnualized": annualized_return(top_group_returns),
        "rebalanceCount": len(ic_series),
        "costRate": cost_rate,
        "applyCost": body.applyCost,
    }

    bench_metrics = None
    if bench_returns:
        bench_cum = 1.0
        for r in bench_returns:
            bench_cum *= (1.0 + r)
        if len(bench_returns) == len(ls_returns):
            bench_alpha_beta = _alpha_beta(bench_returns, ls_returns)
        else:
            bench_alpha_beta = {"alpha": 0.0, "beta": 0.0}
        bench_metrics = {
            "code": bench_code,
            "cumulativeReturn": bench_cum - 1.0,
            "annualizedReturn": annualized_return(bench_returns),
            "annualizedVolatility": annualized_volatility(bench_returns),
            "sharpe": sharpe_ratio(bench_returns),
            "maxDrawdown": max_drawdown(_equity_curve(bench_returns)),
            **bench_alpha_beta,
        }

    ic_stats = information_coefficient_stats([p["ic"] for p in ic_series])
    mean_rank_ic = mean([p["rankIc"] for p in ic_series])

    return {
        "factorLabel": FACTORS[body.factor]["label"], "groups": groups, "n": n,
        "meanIc": ic_stats["meanIc"], "meanRankIc": mean_rank_ic,
        "icWinRate": ic_stats["icWinRate"], "icIr": ic_stats["icIr"],
        "icSeries": ic_series, "groupSummary": group_summary, "longShort": long_short_points,
        "metrics": metrics, "benchmark": bench_metrics,
        "universeSize": len(pool), "effectiveStocks": len(series), "rebalanceCount": len(ic_series),
        "config": body.model_dump(),
    }


def _equity_curve(returns: list[float]) -> list[float]:
    eq = [1.0]
    for r in returns:
        eq.append(eq[-1] * (1.0 + r))
    return eq


def _alpha_beta(bench: list[float], strat: list[float]) -> dict:
    n = len(bench)
    if n < 2:
        return {"alpha": 0.0, "beta": 0.0}
    fit = ols_regression_local(bench, strat)
    return {"alpha": fit["a"], "beta": fit["b"]}


def ols_regression_local(xs: list[float], ys: list[float]) -> dict:
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den = sum((x - mx) ** 2 for x in xs)
    b = 0.0 if den == 0 else num / den
    a = my - b * mx
    return {"a": a, "b": b}


class FactorRegressionBody(BaseModel):
    board: str = "all"
    poolSize: int = 60
    factors: list[str] = ["momentum", "ma_dev"]
    n: int = 5
    hist: int = 180


@router.post("/factor-regression")
async def run_factor_regression(body: FactorRegressionBody):
    """多因子横截面回归（Fama-MacBeth 风格）：每个再平衡日用截面因子暴露对未来收益做多元回归，
    得到各因子的时序收益率，据此评估因子长期是否显著有效（均值/t 统计量/胜率）。"""
    keys = list(dict.fromkeys(k for k in body.factors if k in FACTORS))
    if len(keys) < 2:
        raise HTTPException(400, "请至少选择2个技术类(量价)因子进行多因子回归")
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
    min_sample = len(keys) * 3 + 5
    if len(series) < min_sample:
        raise HTTPException(422, "有效股票样本不足，请增大候选池规模")

    calcs = {k: FACTORS[k]["calc"] for k in keys}
    date_maps = {code: {row["date"]: idx for idx, row in enumerate(kl)} for code, kl in series.items()}
    ref_code = max(series, key=lambda c: len(series[c]))
    ref_dates = [row["date"] for row in series[ref_code]]

    periods = []
    for t in range(25, len(ref_dates) - n, max(1, n)):
        date_t = ref_dates[t]
        xs_rows, ys = [], []
        for code, kl in series.items():
            i = date_maps[code].get(date_t)
            if i is None or i < 20 or i + n >= len(kl):
                continue
            fvs = [calcs[k](kl, i) for k in keys]
            if any(v is None for v in fvs):
                continue
            closes = [row["close"] for row in kl]
            if closes[i] == 0:
                continue
            fret = closes[i + n] / closes[i] - 1
            xs_rows.append(fvs)
            ys.append(fret)
        if len(xs_rows) < min_sample:
            continue

        z_cols = [zscore([r[c] for r in xs_rows]) for c in range(len(keys))]
        z_rows = list(zip(*z_cols))
        fit = multi_ols([list(r) for r in z_rows], ys)
        periods.append({
            "date": date_t, "coefs": dict(zip(keys, fit["coefs"])),
            "intercept": fit["intercept"], "r2": fit["r2"], "sample": len(ys),
        })

    if not periods:
        raise HTTPException(422, "有效截面样本不足，无法完成多因子回归")

    summary = []
    for k in keys:
        vals = [p["coefs"][k] for p in periods]
        m = mean(vals)
        s = std(vals) if len(vals) > 1 else 0.0
        t_stat = 0.0 if s == 0 else m / (s / math.sqrt(len(vals)))
        pos_rate = sum(1 for v in vals if v > 0) / len(vals)
        summary.append({
            "key": k, "label": FACTORS[k]["label"],
            "meanReturn": m, "tStat": t_stat, "positiveRate": pos_rate,
        })

    return {
        "keys": keys, "n": n, "rebalanceCount": len(periods),
        "meanR2": mean([p["r2"] for p in periods]),
        "summary": summary, "periods": periods,
        "universeSize": len(pool), "effectiveStocks": len(series),
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
