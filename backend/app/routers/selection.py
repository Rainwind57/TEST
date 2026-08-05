"""全市场选股：多因子加权打分选股 + 因子分层回测 + 一键落地到自选/模拟盘。"""
import asyncio
import datetime
import logging
import math
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from .. import adapters, db, ml, backtest_event
from .auth import require_user_id
from ..factors import (
    FACTORS, SNAPSHOT_FACTORS, snapshot_factor_value, composite_score,
    bucket_index, pearson, spearman, mean, std, zscore, multi_ols,
    annualized_return, annualized_volatility, sharpe_ratio, sortino_ratio,
    max_drawdown, calmar_ratio, win_rate, information_coefficient_stats,
    round_trip_cost_rate, compute_user_factor_scores,
)
from ..numpy_factors import kline_to_arrays, compute_factor_series, series_at
from .portfolio import OrderBody, place_order

router = APIRouter(prefix="/api/select", tags=["select"])

BOARD_LABELS = {
    "all": "全部A股", "sh_main": "沪市主板", "sz_main": "深市主板",
    "gem": "创业板", "star": "科创板", "bse": "北交所",
    "hs300": "沪深300", "zz500": "中证500", "etf": "ETF基金",
}

BENCHMARKS = {
    "none": None,
    "hs300": "sh000300",
    "zz500": "sh000905",
    "sse": "sh000001",
}


def _price_limit_ratio(code: str, is_st: bool) -> float:
    """A 股涨跌停幅度：创业板(sz30)/科创板(sh68)±20%、北交所(bj)±30%、ST±5%、主板±10%。"""
    if is_st:
        return 0.05
    if code.startswith("sz30") or code.startswith("sh68"):
        return 0.20
    if code.startswith("bj"):
        return 0.30
    return 0.10


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
    factors: list[FactorSpec] = []   # 旧版固定规格加权（key/weight/direction）
    expression: str | None = None    # P1-6 表达式引擎：传表达式时优先于 factors
    topN: int = 20
    filters: FilterSpec = FilterSpec()
    startDate: str | None = None     # 选股时间区间下界（YYYY-MM-DD），不传则用最新截面
    endDate: str | None = None       # 选股时间区间上界（YYYY-MM-DD）
    codes: list[str] | None = None   # 自定义股票池（传 code 列表则跳过 market_list 拉取）
    modelId: str | None = None       # ML 模型 ID：指定时用模型预测分选股（与 factors 二选一）
    adjustId: str | None = None      # 调参配置 artifact id（配合 modelId 使用）
    adjust: dict | None = None       # 或直接传 {featureWeights, threshold}
    assetClass: str = "a-share"      # a-share | future（仅 modelId 分支生效）
    saveArtifact: bool = False       # 选股结果落盘为中间结果，供下一环节（回测/组合/风险）复用


async def _fetch_technical_values(codes: list[str], keys: list[str], hist: int = 500,
                                end_date: str | None = None,
                                start_date: str | None = None) -> dict:
    # 并发上限随池大小自适应：旧版固定 15，大池(500/1000/3000)下分批过多必超时
    sem = asyncio.Semaphore(min(50, max(15, len(codes))))

    async def one(code):
        async with sem:
            try:
                kline = await adapters.fetch_kline(code, hist)
            except Exception as e:
                logger.warning("选股因子计算拉取K线失败 code=%s: %s", code, e)
                return code, {}
            if len(kline) < 25:
                return code, {}
            # 时间区间过滤：指定 startDate/endDate 时在区间内取最后一个截面
            if start_date or end_date:
                filtered = kline
                if start_date:
                    filtered = [r for r in filtered if r["date"] >= start_date]
                if end_date:
                    filtered = [r for r in filtered if r["date"] <= end_date]
                if len(filtered) < 25:
                    return code, {}
                i = len(filtered) - 1
                vals = {key: FACTORS[key]["calc"](filtered, i) for key in keys if key in FACTORS}
            else:
                i = len(kline) - 1
                vals = {key: FACTORS[key]["calc"](kline, i) for key in keys if key in FACTORS}
            return code, vals

    results = await asyncio.gather(*(one(c) for c in codes))
    return dict(results)


@router.post("")
async def run_select(body: SelectBody):
    # ML 模型选股：指定 modelId 时直接用模型预测分排名（打通 ML→选股→模拟盘/盯盘）
    if body.modelId:
        adjust = body.adjust or None
        if body.adjustId and not adjust:
            from .. import artifacts as _artifacts
            rec = _artifacts.load_artifact(body.adjustId)
            if not rec:
                raise HTTPException(404, f"调参配置不存在: {body.adjustId}")
            adjust = rec.get("payload")
        try:
            rows = await ml.score_latest(body.modelId, body.board, body.poolSize,
                                         adjust=adjust, asset_class=body.assetClass)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(422, str(e))
        except Exception as e:
            raise HTTPException(502, f"ML 打分失败: {e}")
        result = {
            "universeSize": len(rows),
            "candidateSize": len(rows),
            "rows": rows[: max(1, body.topN)],
            "modelId": body.modelId,
        }
        if body.saveArtifact:
            from .. import artifacts
            meta = artifacts.save_artifact("select", {
                "codes": [r["code"] for r in rows[: max(1, body.topN)]],
                "rows": rows[: max(1, body.topN)],
                "config": body.model_dump(),
            }, name=f"ML选股-{body.modelId}")
            result["artifact"] = meta
        return result

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
        uf_defs[s.key] = uf

    # 自定义股票池：传 codes 时跳过 market_list 拉取，直接用指定代码
    if body.codes:
        pool = [{"code": c, "name": c} for c in body.codes]
    elif body.board in ("hs300", "zz500"):
        index_map = {"hs300": "sh000300", "zz500": "sh000905"}
        constituents = await adapters.fetch_index_constituents(index_map[body.board])
        if not constituents:
            raise HTTPException(502, f"获取{body.board}成分股失败")
        pool = [{"code": c, "name": c} for c in constituents[:body.poolSize]]
    else:
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

    # 行业因子：勾选时拉取申万一级行业映射，填充到每只候选股
    if "sector" in snap_keys:
        try:
            sector_map = await adapters.fetch_sector_map()
            for row in candidates:
                row["sector"] = sector_map.get(row["code"], "")
        except Exception:
            for row in candidates:
                row["sector"] = ""

    tech_values = {}
    if tech_keys:
        tech_values = await _fetch_technical_values([c["code"] for c in candidates], tech_keys,
                                                     end_date=body.endDate,
                                                     start_date=body.startDate)

    # 资金流/财务快照因子：行情快照不含，勾选时逐股批量拉取（并发 10）
    EXTRA_FINANCE = {"roe", "net_margin", "revenue_yoy", "profit_yoy",
                     "gross_margin", "debt_ratio", "eps", "bps", "roa"}
    EXTRA_MONEYFLOW = {"main_net_pct", "north_holding_pct"}
    need_finance = bool(set(snap_keys) & EXTRA_FINANCE)
    need_mf = bool(set(snap_keys) & EXTRA_MONEYFLOW)
    if need_finance or need_mf:
        sem2 = asyncio.Semaphore(10)

        async def fetch_extra(row):
            code = row["code"]
            async with sem2:
                if need_finance:
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
                if need_mf:
                    try:
                        mf = await adapters.fetch_money_flow(code)
                    except Exception:
                        mf = {}
                    row["main_net_pct"] = mf.get("mainNetPct")
                    # 个股北向持股比例（新增数据源，旧版只有市场级趋势无个股因子）
                    if "north_holding_pct" in snap_keys:
                        try:
                            nh = await adapters.fetch_north_holding(code)
                        except Exception:
                            nh = {}
                        row["north_holding_pct"] = nh.get("holdRatio")

        await asyncio.gather(*(fetch_extra(c) for c in candidates))

    scored_rows = []
    for row in candidates:
        entry = dict(row)
        for key in snap_keys:
            entry[key] = snapshot_factor_value(row, key)
        for key in tech_keys:
            entry[key] = tech_values.get(row["code"], {}).get(key)
        scored_rows.append(entry)

    for uf_key, uf in uf_defs.items():
        definition = uf["definition"] or {}
        if uf["kind"] == "expression":
            # 表达式因子：用安全 AST 引擎对已计算出的因子列求值（momentum - volatility*0.5 等）
            from .. import factor_expr
            expr = definition.get("expression") or ""
            scores = factor_expr.evaluate_expression(expr, scored_rows) if expr else [0.0] * len(scored_rows)
        else:
            scores = compute_user_factor_scores(scored_rows, definition)
        for idx, row in enumerate(scored_rows):
            row[uf_key] = scores[idx]

    # P1-6 表达式引擎：传 expression 时用安全 AST 求值，替代固定规格 composite_score
    from .. import factor_expr
    if body.expression:
        scores = factor_expr.evaluate_expression(body.expression, scored_rows)
        for idx, row in enumerate(scored_rows):
            row["score"] = scores[idx]
            row["factorDetail"] = {"expression": body.expression}
        scored = scored_rows
    else:
        specs = [{"key": s.key, "weight": s.weight, "direction": s.direction} for s in body.factors]
        scored = composite_score(scored_rows, specs)
    scored.sort(key=lambda r: r["score"], reverse=True)
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank

    result = {
        "universeSize": len(pool), "candidateSize": len(candidates),
        "rows": scored[: max(1, body.topN)],
    }
    # 选股结果落盘：供"选股→回测/组合/风险"下一环节读取 codes 复用
    if body.saveArtifact:
        from .. import artifacts
        top_codes = [r["code"] for r in scored[: max(1, body.topN)]]
        meta = artifacts.save_artifact("select", {
            "codes": top_codes,
            "rows": scored[: max(1, body.topN)],
            "config": body.model_dump(),
        }, name=f"选股-{body.board}")
        result["artifact"] = meta
    return result


class BacktestBody(BaseModel):
    board: str = "all"
    poolSize: int = 60
    factor: str = "momentum"
    modelId: str | None = None      # ML 模型 ID：指定时走 ML 信号分层回测（与技术因子二选一）
    groups: int = 5
    n: int = 5
    hist: int = 180
    commissionRate: float = 0.00025   # 佣金费率（万 2.5，单边）
    stampDuty: float = 0.001          # 印花税（千 1，卖出单边）
    slippage: float = 0.001           # 滑点/冲击成本（单边）
    benchmark: str = "none"           # none | hs300 | zz500 | sse
    applyCost: bool = True
    startDate: str | None = None      # 调仓日下界（YYYY-MM-DD，含），用于 IS/OOS 不重叠切分
    endDate: str | None = None        # 调仓日上界（YYYY-MM-DD，含）
    codes: list[str] | None = None    # 自定义股票池（传 code 列表则跳过 market_list 拉取）
    assetClass: str = "a-share"       # a-share | future（期货取主力连续合约池，无涨跌停约束）
    saveArtifact: bool = False        # 回测结果落盘为中间结果，供组合/风险环节复用


@router.post("/backtest")
async def run_backtest(body: BacktestBody):
    # 模型策略：指定 modelId 时走 ML 信号分层回测，响应结构与技术因子回测一致，
    # 前端图表零成本复用（打通”主回测页导入模型”，旧版 modelId 无门可入）。
    if body.modelId:
        try:
            res = await ml.backtest_model(
                body.modelId, body.board, body.poolSize, body.groups, body.n, body.hist,
                body.commissionRate, body.stampDuty, body.slippage, body.benchmark, body.applyCost,
                asset_class=body.assetClass,
                start_date=body.startDate, end_date=body.endDate,
                config=body.model_dump(),
            )
            res["config"] = body.model_dump()  # 补 config 供前端导出报告/保存策略复用
            if body.saveArtifact:
                from .. import artifacts
                meta = artifacts.save_artifact("backtest", res,
                                               name=f"ML回测-{body.modelId}")
                res["artifact"] = meta
            return res
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(422, str(e))
        except Exception as e:
            raise HTTPException(502, f"ML 回测失败: {e}")
    if body.factor not in FACTORS:
        raise HTTPException(400, "分层回测目前仅支持技术类(量价)因子")
    groups = max(2, min(10, body.groups))
    n = max(1, body.n)
    hist = max(60, body.hist)

    bench_code = BENCHMARKS.get(body.benchmark)
    if body.benchmark != "none" and bench_code is None:
        raise HTTPException(400, f"未知基准: {body.benchmark}")

    try:
        if body.codes:
            pool = [{"code": c, "name": c} for c in body.codes]
        elif body.assetClass == "future":
            # 期货回测：候选池取主力连续合约，K线走期货适配器，无涨跌停/ST 约束
            pool = [{"code": c, "name": c} for c in adapters.FUTURE_UNIVERSE[:body.poolSize]]
        else:
            pool = await adapters.fetch_market_list(body.board, body.poolSize)
    except Exception as e:
        raise HTTPException(502, f"候选池获取失败: {e}")

    codes = [row["code"] for row in pool]
    is_st = {} if body.assetClass == "future" else {
        r["code"]: ("ST" in r.get("name", "") or "*ST" in r.get("name", ""))
        for r in pool
    }
    # 回测 K 线并发随池大小自适应（旧版固定 15）
    sem = asyncio.Semaphore(min(50, max(15, len(codes))))

    async def fetch_one(code):
        async with sem:
            try:
                if body.assetClass == "future":
                    kline = await adapters.fetch_future_kline(code, hist)
                else:
                    kline = await adapters.fetch_kline(code, hist)
            except Exception as e:
                logger.warning("回测拉取K线失败 code=%s: %s", code, e)
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

    # 向量化预计算：每只股票一次性算出因子全序列 + close 数组，截面循环只查表
    code_cache = {}
    for code, kl in series.items():
        arr = kline_to_arrays(kl)
        fseries = compute_factor_series(body.factor, arr)
        code_cache[code] = {"closes": arr["close"], "volumes": arr["volume"], "fseries": fseries, "kline": kl}

    cost_rate = round_trip_cost_rate(body.commissionRate, body.stampDuty, body.slippage) if body.applyCost else 0.0

    ic_series = []
    bucket_returns = [[] for _ in range(groups)]
    long_short_points = []
    top_group_returns = []
    bench_by_date = {}
    cum = 1.0
    cum_top = 1.0

    bench_date_idx = {}
    if bench_series:
        bench_date_idx = {row["date"]: idx for idx, row in enumerate(bench_series)}

    # 回测起点 60：与 ML 训练截面 i=60 对齐（旧版 t=25 致 momentum60/dist_52w_high 等
    # 长窗口因子在早期被整股丢弃，前后期有效特征集不一致）
    _WARMUP = 60
    for t in range(_WARMUP, len(ref_dates) - n, max(1, n)):
        date_t = ref_dates[t]
        if body.startDate and date_t < body.startDate:
            continue
        if body.endDate and date_t > body.endDate:
            continue
        cross = []
        for code in series:
            i = date_maps[code].get(date_t)
            if i is None or i < 20 or i + n >= len(code_cache[code]["kline"]):
                continue
            cc = code_cache[code]
            fseries = cc["fseries"]
            fv = series_at(fseries, i) if fseries is not None else calc(cc["kline"], i)
            closes = cc["closes"]
            if fv is None or closes[i] == 0:
                continue
            # 涨跌停约束：t 日涨停封板买不进；t+n 日跌停卖不出（A 股不可强制成交；期货无涨跌停跳过）
            if body.assetClass == "future":
                limit = None
            else:
                limit = _price_limit_ratio(code, is_st.get(code, False))
            if limit is not None:
                if i >= 1 and closes[i - 1] != 0 and closes[i] / closes[i - 1] - 1.0 >= limit - 1e-4:
                    continue
                if closes[i + n - 1] != 0 and closes[i + n] / closes[i + n - 1] - 1.0 <= -limit + 1e-4:
                    continue
            # 停牌处理：持仓期 [i, i+n] 内有停牌日（成交量=0）则跳过该股该期，
            # 旧版跨停牌用缺口价算收益致收益失真（停牌期间价格冻结，收益恒 0 或跳空失真）
            vols = cc["volumes"]
            if any(vols[j] == 0 for j in range(i, i + n + 1)):
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
            # 多空组合 = 做多 top + 做空 bottom，双腿各一份往返成本（旧版仅扣一份）
            net_ls = ls_ret - 2.0 * cost_rate
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
                    bench_by_date[date_t] = bc[bi + n] / bc[bi] - 1

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

    # 调仓间隔 n 日 → 年化采样数 = 252/n（旧版误用 252，年化收益高估 n 倍、Sharpe 高估 √n 倍）
    ppy = 252.0 / max(1, n)

    metrics = {
        "cumulativeReturn": cum - 1.0,
        "annualizedReturn": annualized_return(ls_returns, ppy),
        "annualizedVolatility": annualized_volatility(ls_returns, ppy),
        "sharpe": sharpe_ratio(ls_returns, periods_per_year=ppy),
        "sortino": sortino_ratio(ls_returns, periods_per_year=ppy),
        "maxDrawdown": max_drawdown(ls_equity),
        "calmar": calmar_ratio(ls_returns, ppy),
        "winRate": win_rate(ls_returns),
        "topGroupCumReturn": cum_top - 1.0,
        "topGroupAnnualized": annualized_return(top_group_returns, ppy),
        "rebalanceCount": len(ic_series),
        "costRate": cost_rate,
        "applyCost": body.applyCost,
    }

    bench_metrics = None
    if bench_by_date:
        # 按日期交集对齐策略与基准收益（旧版长度不等即置 alpha/beta=0，掩盖真实 beta）
        aligned = [(p["longShort"], bench_by_date[p["date"]])
                   for p in long_short_points if p["date"] in bench_by_date]
        aligned_ls = [a for a, _ in aligned]
        aligned_bench = [b for _, b in aligned]
        bench_cum = 1.0
        for r in aligned_bench:
            bench_cum *= (1.0 + r)
        bench_alpha_beta = (_alpha_beta(aligned_bench, aligned_ls)
                            if len(aligned_bench) > 1 else {"alpha": 0.0, "beta": 0.0})
        bench_metrics = {
            "code": bench_code,
            "cumulativeReturn": bench_cum - 1.0,
            "annualizedReturn": annualized_return(aligned_bench, ppy),
            "annualizedVolatility": annualized_volatility(aligned_bench, ppy),
            "sharpe": sharpe_ratio(aligned_bench, periods_per_year=ppy),
            "maxDrawdown": max_drawdown(_equity_curve(aligned_bench)),
            **bench_alpha_beta,
        }

    ic_stats = information_coefficient_stats([p["ic"] for p in ic_series], ppy)
    mean_rank_ic = mean([p["rankIc"] for p in ic_series])

    result = {
        "factorLabel": FACTORS[body.factor]["label"], "groups": groups, "n": n,
        "meanIc": ic_stats["meanIc"], "meanRankIc": mean_rank_ic,
        "icWinRate": ic_stats["icWinRate"], "icIr": ic_stats["icIr"],
        "icSeries": ic_series, "groupSummary": group_summary, "longShort": long_short_points,
        "metrics": metrics, "benchmark": bench_metrics,
        "universeSize": len(pool), "effectiveStocks": len(series), "rebalanceCount": len(ic_series),
        "config": body.model_dump(),
        "survivorshipBiasWarning": "候选池为当前上市股票快照，已退市股票不在回测池中，历史收益可能系统性高估",
    }
    if body.saveArtifact:
        from .. import artifacts
        meta = artifacts.save_artifact("backtest", result, name=f"回测-{body.factor}")
        result["artifact"] = meta
    # P10：回测完成自动存档（生成 HTML 报告 + 登记历史记录），失败不阻塞主流程
    try:
        from .. import reporting
        reporting.store_backtest_report(result, config=body.model_dump())
    except Exception:
        pass
    return result


class EventBacktestBody(BaseModel):
    """事件驱动回测（P1-9）：具体下单时序 + 撮合 + 流动性约束。

    与分层回测互补：分层评价因子，事件驱动验证策略的可实盘性。
    signal: momentum_breakout（动量突破买入、回落卖出）| mean_reversion（均值回归）
    """
    board: str = "all"
    poolSize: int = 20
    signal: str = "momentum_breakout"
    factor: str = "momentum"
    threshold: float = 0.05  # 信号阈值
    hist: int = 180
    initialCash: float = 1_000_000.0
    commissionRate: float = 0.00025
    stampDuty: float = 0.001
    slippage: float = 0.001
    applyCost: bool = True
    maxVolumePct: float = 0.10
    maxPositionPct: float = 0.20
    tradeQty: int = 1000


@router.post("/backtest-event")
async def run_event_backtest(body: EventBacktestBody):
    """事件驱动回测：逐 bar 撮合 + T+1 + 涨跌停 + 流动性约束（P1-9）。"""
    try:
        pool = await adapters.fetch_market_list(body.board, body.poolSize)
    except Exception as e:
        raise HTTPException(502, f"候选池获取失败: {e}")
    codes = [r["code"] for r in pool]
    sem = asyncio.Semaphore(min(50, max(15, len(codes))))
    kline_by_code = {}

    async def fetch_one(code):
        async with sem:
            try:
                kl = await adapters.fetch_kline(code, body.hist)
                return code, kl
            except Exception:
                return code, []
    fetched = await asyncio.gather(*(fetch_one(c) for c in codes))
    for code, kl in fetched:
        if len(kl) >= 40:
            kline_by_code[code] = kl
    if len(kline_by_code) < 3:
        raise HTTPException(422, "有效股票样本不足，需≥3 只且历史≥40 日")

    cfg = backtest_event.EventBacktestConfig(
        initial_cash=body.initialCash, commission_rate=body.commissionRate,
        stamp_duty=body.stampDuty, slippage=body.slippage, apply_cost=body.applyCost,
        max_volume_pct=body.maxVolumePct, max_position_pct=body.maxPositionPct,
    )
    bt = backtest_event.EventBacktest(cfg, kline_by_code)

    # 预计算每只股票因子序列
    factor_series = {}
    for code in kline_by_code:
        arr = kline_to_arrays(kline_by_code[code])
        factor_series[code] = compute_factor_series(body.factor, arr)

    def signal_fn(date, prev, state):
        orders = []
        for code in kline_by_code:
            fseries = factor_series.get(code)
            if fseries is None:
                continue
            idx = bt._date_idx_by_code[code].get(date)
            if idx is None:
                continue
            val = series_at(fseries, idx)
            if val is None:
                continue
            pos = state["positions"].get(code, {})
            cur_qty = pos.get("qty", 0)
            if body.signal == "momentum_breakout":
                if val > body.threshold and cur_qty == 0:
                    orders.append(backtest_event.Order(code=code, side="buy", qty=body.tradeQty))
                elif val < -body.threshold and cur_qty > 0:
                    orders.append(backtest_event.Order(code=code, side="sell", qty=cur_qty))
            elif body.signal == "mean_reversion":
                if val < -body.threshold and cur_qty == 0:
                    orders.append(backtest_event.Order(code=code, side="buy", qty=body.tradeQty))
                elif val > body.threshold and cur_qty > 0:
                    orders.append(backtest_event.Order(code=code, side="sell", qty=cur_qty))
        return orders

    return bt.run(signal_fn)


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
    sem = asyncio.Semaphore(min(50, max(15, len(codes))))

    async def fetch_one(code):
        async with sem:
            try:
                kline = await adapters.fetch_kline(code, hist)
            except Exception as e:
                logger.warning("回测拉取K线失败 code=%s: %s", code, e)
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

    # 向量化预计算：每只股票每因子一次性算出全序列
    fr_cache = {}
    for code, kl in series.items():
        arr = kline_to_arrays(kl)
        fr_cache[code] = {
            "closes": arr["close"],
            "volumes": arr["volume"],
            "series": {k: compute_factor_series(k, arr) for k in keys},
            "kline": kl,
        }

    periods = []
    for t in range(60, len(ref_dates) - n, max(1, n)):
        date_t = ref_dates[t]
        xs_rows, ys = [], []
        for code in series:
            i = date_maps[code].get(date_t)
            if i is None or i < 20 or i + n >= len(fr_cache[code]["kline"]):
                continue
            cc = fr_cache[code]
            fvs = []
            ok = True
            for k in keys:
                s = cc["series"][k]
                v = series_at(s, i) if s is not None else calcs[k](cc["kline"], i)
                if v is None:
                    ok = False
                    break
                fvs.append(v)
            if not ok:
                continue
            closes = cc["closes"]
            if closes[i] == 0:
                continue
            # 停牌处理：持仓期内有停牌日则跳过，避免缺口价算收益失真
            vols = cc["volumes"]
            if any(vols[j] == 0 for j in range(i, i + n + 1)):
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
async def apply_selection(body: ApplyBody, uid: int = Depends(require_user_id)):
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
                await place_order(OrderBody(code=code, side="buy", qty=qty), uid)
                results.append({"code": code, "ok": True, "qty": qty})
            except HTTPException as e:
                results.append({"code": code, "ok": False, "reason": e.detail})
        return {"ok": True, "results": results}

    raise HTTPException(400, "action 必须为 watchlist 或 buy")
