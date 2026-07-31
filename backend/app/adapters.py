"""行情与历史K线数据源适配器（服务端直连，无跨域问题）。"""
import asyncio
import json
import re
import time
import datetime
import httpx
from collections import OrderedDict

from . import db

_kline_cache: OrderedDict[str, tuple[float, list]] = OrderedDict()
KLINE_CACHE_TTL = 300  # 秒
KLINE_CACHE_MAX = 500  # LRU 上限，防内存无限增长


def _kline_cache_set(key: str, value: tuple[float, list]) -> None:
    """写入缓存并按 LRU 淘汰最旧条目。"""
    _kline_cache[key] = value
    _kline_cache.move_to_end(key)
    while len(_kline_cache) > KLINE_CACHE_MAX:
        _kline_cache.popitem(last=False)


async def _http_get(url: str, timeout: int = 10, headers: dict | None = None,
                    retries: int = 2, base_delay: float = 0.5) -> httpx.Response:
    """带指数退避重试的 HTTP GET（腾讯/东财接口偶发限流，旧版无重试静默丢样本）。"""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout, headers=headers or {}) as client:
                return await client.get(url)
        except Exception as e:
            last_exc = e
            if attempt < retries:
                await asyncio.sleep(base_delay * (2 ** attempt))
    raise last_exc


def _to_secid(code: str) -> str:
    return ("1" if code.startswith("sh") else "0") + "." + code[2:]


async def fetch_tencent_quotes(codes: list[str]) -> dict:
    """腾讯实时行情：稳定，带 CORS，字段丰富（含 PE/PB/市值/换手率）。"""
    url = f"https://qt.gtimg.cn/q={','.join(codes)}"
    resp = await _http_get(url, 10)
    text = resp.content.decode("gbk", errors="ignore")
    out = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("v_"):
            continue
        try:
            key, raw = line.split("=", 1)
            code = key[2:].strip()
            raw = raw.strip().strip('"').strip(";").strip('"')
        except ValueError:
            continue
        out[code] = _parse_tencent(raw)
    return out


def _parse_tencent(raw: str) -> dict:
    a = raw.split("~")

    def f(i, default=None):
        try:
            return float(a[i])
        except (IndexError, ValueError):
            return default

    dt = a[30] if len(a) > 30 else ""
    date = f"{dt[0:4]}-{dt[4:6]}-{dt[6:8]}" if len(dt) >= 8 else ""
    tm = f"{dt[8:10]}:{dt[10:12]}:{dt[12:14]}" if len(dt) >= 14 else ""
    # 价格/量额字段缺失用 0（有业务意义：停牌/异常）；pe/pb/turnover/市值缺失用 None（与真实 0 区分）
    return {
        "name": a[1] if len(a) > 1 else "",
        "code": a[2] if len(a) > 2 else "",
        "price": f(3) or 0.0, "preClose": f(4) or 0.0, "open": f(5) or 0.0,
        "volume": (f(6) or 0) * 100,
        "bid": f(9), "ask": f(19),
        "high": f(33), "low": f(34),
        "amount": (f(37) or 0) * 10000,
        "turnover": f(38),
        "pe": f(39),
        "mktCap": f(44),
        "circMktCap": f(45),
        "pb": f(46),
        "date": date, "time": tm,
    }


async def fetch_sina_quotes(codes: list[str]) -> dict:
    """新浪行情：需要带 Referer 规避防盗链。"""
    url = f"https://hq.sinajs.cn/list={','.join(codes)}"
    headers = {"Referer": "https://finance.sina.com.cn/"}
    resp = await _http_get(url, 10, headers=headers)
    text = resp.content.decode("gbk", errors="ignore")
    out = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("var hq_str_"):
            continue
        try:
            key, raw = line.split("=", 1)
            code = key.replace("var hq_str_", "").strip()
            raw = raw.strip().strip(";").strip('"')
        except ValueError:
            continue
        if not raw:
            continue
        out[code] = _parse_sina(raw)
    return out


def _parse_sina(raw: str) -> dict:
    a = raw.split(",")

    def f(i, default=None):
        try:
            return float(a[i])
        except (IndexError, ValueError):
            return default

    # 新浪行情无 pe/pb/turnover/市值字段（需走快照接口）；价格缺失用 0，其余 None
    return {
        "name": a[0] if a else "", "open": f(1) or 0.0, "preClose": f(2) or 0.0, "price": f(3) or 0.0,
        "high": f(4), "low": f(5), "bid": f(6), "ask": f(7),
        "volume": f(8) or 0, "amount": f(9) or 0,
        "date": a[30] if len(a) > 30 else "", "time": a[31] if len(a) > 31 else "",
    }


async def fetch_eastmoney_quotes(codes: list[str]) -> dict:
    """东方财富行情：JSON 格式，价格字段需 /100。"""
    secids = ",".join(_to_secid(c) for c in codes)
    fields = "f12,f13,f14,f43,f44,f45,f46,f47,f48,f60"
    url = f"https://push2.eastmoney.com/api/qt/ulist.np/get?secids={secids}&fields={fields}"
    resp = await _http_get(url, 10)
    data = resp.json()
    out = {}
    diff = (data or {}).get("data", {}).get("diff") or []
    items = diff.values() if isinstance(diff, dict) else diff
    for item in items:
        mkt = "sh" if item.get("f13") == 1 else "sz"
        code = mkt + str(item.get("f12", ""))
        out[code] = _parse_eastmoney(item)
    return out


def _parse_eastmoney(item: dict) -> dict:
    def p(key):
        v = item.get(key)
        if v in (None, "", "-"):
            return None
        try:
            return float(v) / 100
        except (TypeError, ValueError):
            return None

    return {
        "name": item.get("f14", ""), "code": str(item.get("f12", "")),
        "price": p("f43") or 0.0, "high": p("f44"), "low": p("f45"),
        "open": p("f46"), "preClose": p("f60") or 0.0,
        "volume": float(item.get("f47", 0) or 0) * 100,
        "amount": float(item.get("f48", 0) or 0),
        "bid": None, "ask": None, "date": "", "time": "",
    }


SOURCES = {
    "tencent": fetch_tencent_quotes,
    "sina": fetch_sina_quotes,
    "eastmoney": fetch_eastmoney_quotes,
}

BOARD_MATCHERS = {
    "all": lambda c: not c.startswith(("900", "200")),
    "sh_main": lambda c: c.startswith(("600", "601", "603", "605")),
    "sz_main": lambda c: c.startswith(("000", "001", "002", "003")),
    "gem": lambda c: c.startswith(("300", "301")),
    "star": lambda c: c.startswith(("688", "689")),
    "bse": lambda c: c.startswith(("8", "920")),
}

_SINA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Referer": "https://finance.sina.com.cn/",
}
_SINA_MARKET_URL = "https://vip.stock.finance.sina.com.cn/quotes_service/api/jsonp.php/x/Market_Center.getHQNodeData"
_SINA_PAGE_SIZE = 100
_SINA_MAX_PAGES = 50  # 最多扫描 5000 只，覆盖全市场

_market_cache: dict[str, tuple[float, list]] = {}
MARKET_CACHE_TTL = 60  # 秒


def _num(item: dict, key: str):
    v = item.get(key)
    if v is None or v == "" or v == "-":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_sina_market_row(item: dict) -> dict:
    mkt_cap = _num(item, "mktcap")  # 万元
    circ_cap = _num(item, "nmc")  # 万元
    return {
        "code": item.get("symbol", ""),
        "name": item.get("name", ""),
        "price": _num(item, "trade"),
        "pctChg": _num(item, "changepercent"),
        "volume": _num(item, "volume"),
        "amount": _num(item, "amount"),
        "turnover": _num(item, "turnoverratio"),
        "pe": _num(item, "per"),
        "pb": _num(item, "pb"),
        "mktCap": None if mkt_cap is None else mkt_cap / 10000,
        "circMktCap": None if circ_cap is None else circ_cap / 10000,
    }


def _parse_sina_jsonp(text: str):
    m = re.search(r"x\((\[.*\])\)", text, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(1))
    except (json.JSONDecodeError, ValueError):
        return []


async def _fetch_sina_page(client: httpx.AsyncClient, page: int, sort_field: str) -> list[dict]:
    params = {
        "page": page, "num": _SINA_PAGE_SIZE, "sort": sort_field, "asc": 0,
        "node": "hs_a", "symbol": "", "_s_r_a": "page",
    }
    resp = await client.get(_SINA_MARKET_URL, params=params)
    return _parse_sina_jsonp(resp.text)


# 东财 clist 板块 → fs 市场范围参数（新浪降级 fallback 用）
_EASTMONEY_FS = {
    "all": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
    "sh_main": "m:1 t:2",
    "sz_main": "m:0 t:6",
    "gem": "m:0 t:80",
    "star": "m:1 t:23",
    "bse": "m:0 t:81 s:2048",
}


def _eastmoney_code(f12: str, f13: int) -> str:
    """东财纯代码+市场标志 → sh/sz/bj 前缀代码（与腾讯 K 线接口一致）。"""
    pure = str(f12 or "")
    if pure.startswith(("8", "920", "4")):
        return "bj" + pure
    return ("sh" if f13 == 1 else "sz") + pure


def _parse_eastmoney_market_row(item: dict) -> dict:
    def g(key, default=None):
        v = item.get(key)
        return default if v in (None, "", "-") else v

    def nf(key):
        v = g(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    mkt = nf("f20")  # 元
    circ = nf("f21")  # 元
    return {
        "code": _eastmoney_code(item.get("f12", ""), item.get("f13", 0)),
        "name": g("f14", ""),
        "price": nf("f2"),
        "pctChg": nf("f3"),
        "volume": nf("f5"),
        "amount": nf("f6"),
        "turnover": nf("f8"),
        "pe": nf("f9"),
        "pb": nf("f23"),
        "mktCap": None if mkt is None else mkt / 1e8,        # 元 → 亿元，与新浪口径一致
        "circMktCap": None if circ is None else circ / 1e8,
    }


async def fetch_eastmoney_market(board: str = "all", limit: int = 300) -> list[dict]:
    """东方财富全市场/分板块行情快照（新浪降级 fallback）。

    单页拉取（pz=limit），字段与 _parse_sina_market_row 对齐；按 board 过滤。
    """
    fs = _EASTMONEY_FS.get(board, _EASTMONEY_FS["all"])
    matcher = BOARD_MATCHERS.get(board, BOARD_MATCHERS["all"])
    limit = max(1, min(limit, 3000))
    url = ("https://push2.eastmoney.com/api/qt/clist/get"
           f"?pn=1&pz={limit}&po=1&np=1&fltt=2&invt=2&fid=f6&fs={fs}"
           f"&fields=f12,f13,f14,f2,f3,f5,f6,f8,f9,f23,f20,f21")
    resp = await _http_get(url, 15)
    rows = (resp.json() or {}).get("data", {}).get("diff") or []
    items = rows.values() if isinstance(rows, dict) else rows
    out = []
    seen: set[str] = set()
    for item in items:
        row = _parse_eastmoney_market_row(item)
        if not row["code"] or row["code"] in seen:
            continue
        seen.add(row["code"])
        if matcher(row["code"][2:]):
            out.append(row)
    return out[:limit]


async def fetch_market_list(board: str = "all", limit: int = 300, sort_field: str = "amount") -> list[dict]:
    """全市场/分板块行情快照。

    主源新浪 Market_Center（沪深A股全量，分页拉取后按代码前缀做板块过滤）；
    新浪限流/失败时降级到东方财富 clist，避免单源故障导致候选池全空。
    """
    matcher = BOARD_MATCHERS.get(board, BOARD_MATCHERS["all"])
    limit = max(1, min(limit, 3000))
    cache_key = f"{board}:{limit}:{sort_field}"
    cached = _market_cache.get(cache_key)
    if cached and time.time() - cached[0] < MARKET_CACHE_TTL:
        return cached[1]

    try:
        rows = await _fetch_sina_market(board, limit, sort_field, matcher)
    except Exception:
        rows = []
    if not rows:
        # 新东财降级：新浪限流/返回空时用东财 clist 兜底，避免“网络一差→股票池全空→全线报样本不足”
        try:
            rows = await fetch_eastmoney_market(board, limit)
        except Exception:
            rows = []

    # 空结果禁止写缓存：旧版把空列表也缓存 60s，故障被放大；空时强制下次重拉
    if rows:
        _market_cache[cache_key] = (time.time(), rows)
    return rows


async def _fetch_sina_market(board: str, limit: int, sort_field: str, matcher) -> list[dict]:
    """新浪 Market_Center 分页拉取 + 板块过滤。失败抛异常由上层降级。"""
    matched: list[dict] = []
    seen_codes: set[str] = set()
    async with httpx.AsyncClient(timeout=15, headers=_SINA_HEADERS) as client:
        page = 1
        batch_size = 5
        while len(matched) < limit and page <= _SINA_MAX_PAGES:
            pages = range(page, min(page + batch_size, _SINA_MAX_PAGES + 1))
            results = await asyncio.gather(*(_fetch_sina_page(client, p, sort_field) for p in pages))
            got_any = False
            for rows in results:
                if rows:
                    got_any = True
                for item in rows:
                    row = _parse_sina_market_row(item)
                    if not row["code"] or row["code"] in seen_codes:
                        continue
                    seen_codes.add(row["code"])
                    if matcher(row["code"][2:]):
                        matched.append(row)
            page += batch_size
            if not got_any:
                break
    return matched[:limit]


async def fetch_quotes(codes: list[str], source: str = "tencent") -> dict:
    fn = SOURCES.get(source, fetch_tencent_quotes)
    return await fn(codes)


async def fetch_kline(code: str, days: int = 150, force_refresh: bool = False) -> list[dict]:
    """历史日K线（腾讯，前复权）。

    三级缓存：内存(L1, 300s) → SQLite 磁盘(L2) → 网络。已有足够历史时直接返回缓存，
    仅在缓存不足或强制刷新时回源，回源后增量写盘。重复回测可减少 90%+ 网络请求。
    """
    cache_key = f"{code}:{days}"
    if not force_refresh:
        cached = _kline_cache.get(cache_key)
        if cached and time.time() - cached[0] < KLINE_CACHE_TTL:
            _kline_cache.move_to_end(cache_key)  # LRU 命中提升
            return cached[1]

        disk_rows = db.get_cached_kline(code)
        if len(disk_rows) >= days and not _kline_stale(disk_rows):
            result = disk_rows[-days:]  # 裁剪到请求长度（旧版静默放大返回全量）
            _kline_cache_set(cache_key, (time.time(), result))
            return result

    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,{days},qfq"
    resp = await _http_get(url, 10)
    data = resp.json()
    stock_data = (data or {}).get("data", {}).get(code, {})
    arr = stock_data.get("qfqday") or stock_data.get("day") or []
    kline = [
        {"date": row[0], "open": float(row[1]), "close": float(row[2]),
         "high": float(row[3]), "low": float(row[4]), "volume": float(row[5])}
        for row in arr
    ]

    # 空结果禁止写缓存：旧版把限流返回的空 K 线也落盘+入内存缓存，此后 300s 内
    # 所有调用都拿到空数据，单次网络抖动被放大成 5 分钟故障。
    if not kline:
        return []
    db.upsert_kline(code, kline)
    merged = db.get_cached_kline(code)
    result = merged[-days:]
    _kline_cache_set(cache_key, (time.time(), result))
    return result


def _kline_stale(rows: list[dict], max_age_days: int = 3) -> bool:
    """K 线末日日期新鲜度校验：末日距今超过 max_age_days 日历日视为陈旧需回源。"""
    if not rows:
        return True
    last_date = rows[-1].get("date", "")
    try:
        last_dt = datetime.datetime.strptime(last_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        return True
    return (datetime.datetime.now() - last_dt).days > max_age_days


# ---------------- 资金流向（东方财富） ----------------

_money_flow_cache: dict[str, tuple[float, dict]] = {}
MONEY_FLOW_CACHE_TTL = 300  # 秒


def _to_secid_full(code: str) -> str:
    """完整 secid：沪市 1.、深市 0.。"""
    market = "1" if code.startswith("sh") else "0"
    return f"{market}.{code[2:]}"


async def fetch_money_flow(code: str) -> dict:
    """单股资金流向（东方财富）：主力/超大单/大单/中单/小单 净额与净占比。"""
    cache_key = code
    cached = _money_flow_cache.get(cache_key)
    if cached and time.time() - cached[0] < MONEY_FLOW_CACHE_TTL:
        return cached[1]

    secid = _to_secid_full(code)
    url = (f"https://push2.eastmoney.com/api/qt/stock/get"
           f"?secid={secid}&fields=f62,f184,f66,f69,f72,f75,f78,f81,f84,f87")
    resp = await _http_get(url, 10)
    data = (resp.json() or {}).get("data", {})
    if not data:
        return {}
    f = lambda k: float(data.get(k, 0) or 0)
    result = {
        "mainNet": f("f62"), "mainNetPct": f("f184") / 100,
        "superLargeNet": f("f66"), "superLargePct": f("f69") / 100,
        "largeNet": f("f72"), "largePct": f("f75") / 100,
        "mediumNet": f("f78"), "mediumPct": f("f81") / 100,
        "smallNet": f("f84"), "smallPct": f("f87") / 100,
    }
    _money_flow_cache[cache_key] = (time.time(), result)
    return result


async def fetch_money_flow_trend(code: str, days: int = 10) -> list[dict]:
    """单股 N 日资金流向趋势（东方财富历史接口）。"""
    market = "1" if code.startswith("sh") else "0"
    pure_code = code[2:]
    url = (f"https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
           f"?secid={market}.{pure_code}&lmt={days}&fields1=f1,f2,f3,f7"
           f"&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65")
    resp = await _http_get(url, 10)
    klines = (resp.json() or {}).get("data", {}).get("klines", [])
    out = []
    for line in klines:
        parts = line.split(",")
        if len(parts) < 7:
            continue
        try:
            out.append({
                "date": parts[0],
                "mainNet": float(parts[1]),
                "smallNet": float(parts[2]),
                "mediumNet": float(parts[3]),
                "largeNet": float(parts[4]),
                "superLargeNet": float(parts[5]),
            })
        except (IndexError, ValueError):
            continue
    return out


# ---------------- 北向资金（东方财富） ----------------

_north_flow_cache: dict[str, tuple[float, list]] = {}
NORTH_FLOW_CACHE_TTL = 300


async def fetch_north_flow_trend(days: int = 30) -> list[dict]:
    """北向资金 N 日净流入趋势（沪股通+深股通合计）。"""
    cache_key = f"north:{days}"
    cached = _north_flow_cache.get(cache_key)
    if cached and time.time() - cached[0] < NORTH_FLOW_CACHE_TTL:
        return cached[1]

    url = (f"https://push2his.eastmoney.com/api/qt/kamt.kline/get"
           f"?fields1=f1,f2,f3,f4&fields2=f51,f52,f53,f54,f55,f56"
           f"&klt=101&lmt={days}")
    resp = await _http_get(url, 10)
    klines = (resp.json() or {}).get("data", {}).get("s2n", [])
    out = []
    for line in klines:
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            out.append({
                "date": parts[0],
                "shNet": float(parts[1]),  # 沪股通净买入（万元）
                "szNet": float(parts[2]),  # 深股通净买入
                "totalNet": float(parts[1]) + float(parts[2]),
            })
        except (IndexError, ValueError):
            continue
    _north_flow_cache[cache_key] = (time.time(), out)
    return out


# ---------------- 财务指标（东方财富） ----------------

async def fetch_finance_summary(code: str) -> dict:
    """主要财务指标摘要（东方财富个股财务接口）：ROE/毛利率/净利率/资产负债率/ROA 等。"""
    market = "1" if code.startswith("sh") else "0"
    pure = code[2:]
    url = (f"https://datacenter.eastmoney.com/securities/api/data/get"
           f"?type=RPT_F10_FINANCE_MAINFINADATA&sty=APP_F10_MAINFINADATA"
           f"&filter=(SECUCODE%3D%22{pure}.{market.upper()}%22)"
           f"&p=1&ps=4&sr=-1&st=REPORT_DATE")
    resp = await _http_get(url, 10)
    rows = (resp.json() or {}).get("result", {}).get("data", [])
    if not rows:
        return {}
    r = rows[0]
    f = lambda k: r.get(k)
    # ROA = 归母净利润 / 总资产（东财接口含 TOTAL_ASSETS / PARENT_NETPROFIT，旧版未拉）
    total_assets = f("TOTAL_ASSETS")
    net_profit = f("PARENT_NETPROFIT")
    roa = None
    if total_assets and net_profit and total_assets != 0:
        try:
            roa = float(net_profit) / float(total_assets) * 100
        except (TypeError, ValueError):
            roa = None
    return {
        "reportDate": f("REPORT_DATE"),
        "roe": f("WEIGHTAVG_ROE"),
        "roa": roa,
        "grossMargin": f("GROSS_PROFIT_RATIO"),
        "netMargin": f("NET_PROFIT_RATIO"),
        "debtRatio": f("DEBT_ASSET_RATIO"),
        "revenueYoY": f("TOTAL_OPERATE_INCOME_YOY"),
        "profitYoY": f("PARENT_NETPROFIT_YOY"),
        "eps": f("BASIC_EPS"),
        "bps": f("BPS"),
    }


async def fetch_north_holding(code: str) -> dict:
    """个股北向持股（东方财富沪深港通持股明细）：持股比例/持股数/市值。

    旧版只有市场级 fetch_north_flow_trend 却无个股北向因子，此处补齐个股级数据源。
    """
    pure = code[2:]
    url = (f"https://datacenter-web.eastmoney.com/api/data/v1/get"
           f"?reportName=RPT_MUTUAL_HOLDSTOCKS_DETAILS&columns=ALL"
           f"&filter=(SECURITY_CODE%3D%22{pure}%22)"
           f"&pageSize=1&sortColumns=REPORT_DATE&sortTypes=-1")
    try:
        resp = await _http_get(url, 10)
        records = (resp.json() or {}).get("result", {}).get("data") or []
        if not records:
            return {}
        r = records[0]
        return {
            "holdRatio": r.get("HOLD_SHARES_RATIO"),     # 持股比例(%)
            "holdShares": r.get("HOLD_SHARES"),           # 持股数(股)
            "holdMarketValue": r.get("HOLD_MARKET_VALUE"),# 持股市值(元)
            "reportDate": r.get("REPORT_DATE"),
        }
    except Exception:
        return {}


# ---------------- 分钟级 K 线（腾讯） ----------------

_minute_cache: dict[str, tuple[float, list]] = {}
MINUTE_CACHE_TTL = 120  # 秒


async def fetch_minute_kline(code: str, period: str = "5", count: int = 240) -> list[dict]:
    """分钟级 K 线（腾讯）。

    period: "1"=1分钟, "5"=5分钟, "15"=15分钟, "30"=30分钟, "60"=60分钟。
    count: K 线数量。默认 240（≈ 一个交易日的 1 分钟数）。
    返回 [{datetime, open, close, high, low, volume}]，按时间升序。
    """
    cache_key = f"{code}:{period}:{count}"
    cached = _minute_cache.get(cache_key)
    if cached and time.time() - cached[0] < MINUTE_CACHE_TTL:
        return cached[1]

    url = (f"https://web.ifzq.gtimg.cn/appstock/app/kline/mkline/get"
           f"?param={code},m{period},{count}")
    resp = await _http_get(url, 10)
    data = (resp.json() or {}).get("data", {}).get(code, {})
    arr = data.get(f"m{period}") or []
    out = []
    for row in arr:
        try:
            out.append({
                "datetime": row[0],
                "open": float(row[1]), "close": float(row[2]),
                "high": float(row[3]), "low": float(row[4]),
                "volume": float(row[5]),
            })
        except (IndexError, ValueError):
            continue
    _minute_cache[cache_key] = (time.time(), out)
    return out
