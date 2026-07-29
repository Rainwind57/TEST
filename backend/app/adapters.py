"""行情与历史K线数据源适配器（服务端直连，无跨域问题）。"""
import asyncio
import json
import re
import time
import httpx

from . import db

_kline_cache: dict[str, tuple[float, list]] = {}
KLINE_CACHE_TTL = 300  # 秒


def _to_secid(code: str) -> str:
    return ("1" if code.startswith("sh") else "0") + "." + code[2:]


async def fetch_tencent_quotes(codes: list[str]) -> dict:
    """腾讯实时行情：稳定，带 CORS，字段丰富（含 PE/PB/市值/换手率）。"""
    url = f"https://qt.gtimg.cn/q={','.join(codes)}"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
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

    def f(i, default=0.0):
        try:
            return float(a[i])
        except (IndexError, ValueError):
            return default

    dt = a[30] if len(a) > 30 else ""
    date = f"{dt[0:4]}-{dt[4:6]}-{dt[6:8]}" if len(dt) >= 8 else ""
    tm = f"{dt[8:10]}:{dt[10:12]}:{dt[12:14]}" if len(dt) >= 14 else ""
    return {
        "name": a[1] if len(a) > 1 else "",
        "code": a[2] if len(a) > 2 else "",
        "price": f(3), "preClose": f(4), "open": f(5),
        "volume": f(6) * 100,
        "bid": f(9), "ask": f(19),
        "high": f(33), "low": f(34),
        "amount": f(37) * 10000,
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
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, headers=headers)
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

    def f(i, default=0.0):
        try:
            return float(a[i])
        except (IndexError, ValueError):
            return default

    return {
        "name": a[0] if a else "", "open": f(1), "preClose": f(2), "price": f(3),
        "high": f(4), "low": f(5), "bid": f(6), "ask": f(7),
        "volume": f(8), "amount": f(9),
        "date": a[30] if len(a) > 30 else "", "time": a[31] if len(a) > 31 else "",
    }


async def fetch_eastmoney_quotes(codes: list[str]) -> dict:
    """东方财富行情：JSON 格式，价格字段需 /100。"""
    secids = ",".join(_to_secid(c) for c in codes)
    fields = "f12,f13,f14,f43,f44,f45,f46,f47,f48,f60"
    url = f"https://push2.eastmoney.com/api/qt/ulist.np/get?secids={secids}&fields={fields}"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
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
        try:
            return float(item.get(key, 0)) / 100
        except (TypeError, ValueError):
            return 0.0

    return {
        "name": item.get("f14", ""), "code": str(item.get("f12", "")),
        "price": p("f43"), "high": p("f44"), "low": p("f45"), "open": p("f46"), "preClose": p("f60"),
        "volume": float(item.get("f47", 0) or 0) * 100,
        "amount": float(item.get("f48", 0) or 0),
        "bid": 0.0, "ask": 0.0, "date": "", "time": "",
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


async def fetch_market_list(board: str = "all", limit: int = 300, sort_field: str = "amount") -> list[dict]:
    """全市场/分板块行情快照（新浪 Market_Center，沪深A股全量，分页拉取后按代码前缀做板块过滤）。"""
    matcher = BOARD_MATCHERS.get(board, BOARD_MATCHERS["all"])
    limit = max(1, min(limit, 3000))
    cache_key = f"{board}:{limit}:{sort_field}"
    cached = _market_cache.get(cache_key)
    if cached and time.time() - cached[0] < MARKET_CACHE_TTL:
        return cached[1]

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

    rows = matched[:limit]
    _market_cache[cache_key] = (time.time(), rows)
    return rows


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
            return cached[1]

        disk_rows = db.get_cached_kline(code)
        if len(disk_rows) >= days:
            _kline_cache[cache_key] = (time.time(), disk_rows)
            return disk_rows

    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,{days},qfq"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
    data = resp.json()
    stock_data = (data or {}).get("data", {}).get(code, {})
    arr = stock_data.get("qfqday") or stock_data.get("day") or []
    kline = [
        {"date": row[0], "open": float(row[1]), "close": float(row[2]),
         "high": float(row[3]), "low": float(row[4]), "volume": float(row[5])}
        for row in arr
    ]

    db.upsert_kline(code, kline)
    merged = db.get_cached_kline(code) if kline else kline
    _kline_cache[cache_key] = (time.time(), merged)
    return merged


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
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
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
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
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
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
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
    """主要财务指标摘要（东方财富个股财务接口）：ROE/毛利率/净利率/资产负债率等。"""
    market = "1" if code.startswith("sh") else "0"
    pure = code[2:]
    url = (f"https://datacenter.eastmoney.com/securities/api/data/get"
           f"?type=RPT_F10_FINANCE_MAINFINADATA&sty=APP_F10_MAINFINADATA"
           f"&filter=(SECUCODE%3D%22{pure}.{market.upper()}%22)"
           f"&p=1&ps=4&sr=-1&st=REPORT_DATE")
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
    rows = (resp.json() or {}).get("result", {}).get("data", [])
    if not rows:
        return {}
    r = rows[0]
    f = lambda k: r.get(k)
    return {
        "reportDate": f("REPORT_DATE"),
        "roe": f("WEIGHTAVG_ROE"),
        "grossMargin": f("GROSS_PROFIT_RATIO"),
        "netMargin": f("NET_PROFIT_RATIO"),
        "debtRatio": f("DEBT_ASSET_RATIO"),
        "revenueYoY": f("TOTAL_OPERATE_INCOME_YOY"),
        "profitYoY": f("PARENT_NETPROFIT_YOY"),
        "eps": f("BASIC_EPS"),
        "bps": f("BPS"),
    }


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
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
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
