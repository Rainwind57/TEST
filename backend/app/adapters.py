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
