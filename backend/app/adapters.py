"""行情与历史K线数据源适配器（服务端直连，无跨域问题）。"""
import asyncio
import json
import re
import threading
import time
import datetime
import httpx
from urllib.parse import urlencode
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


# 共享 HTTP 客户端（按事件循环隔离）：复用 TCP/TLS 连接池，避免每次请求重建握手。
# 旧版 _http_get 每次调用都新建 AsyncClient，选股/回测/ML 对成百上千只股票各拉一次
# K 线时握手开销被成倍放大，是"十分缓慢"的主因。httpx 客户端绑定创建时的 event loop，
# 故按 loop id 隔离（optimize 用独立临时 loop 跑回测）。
_http_clients: dict[int, httpx.AsyncClient] = {}
_clients_lock = threading.Lock()


async def _http_client() -> httpx.AsyncClient:
    """返回当前事件循环上的共享 AsyncClient（懒创建）。"""
    loop_id = id(asyncio.get_running_loop())
    with _clients_lock:
        client = _http_clients.get(loop_id)
        if client is None:
            client = httpx.AsyncClient(timeout=10, follow_redirects=True)
            _http_clients[loop_id] = client
        return client


async def close_http_client() -> None:
    """关闭当前事件循环上注册的共享客户端（应用关停 / optimize 临时 loop 结束时调用）。"""
    loop_id = id(asyncio.get_running_loop())
    with _clients_lock:
        client = _http_clients.pop(loop_id, None)
    if client is not None:
        await client.aclose()


async def _http_get(url: str, timeout: int = 8, headers: dict | None = None,
                    retries: int = 2, base_delay: float = 0.5) -> httpx.Response:
    """带指数退避重试的 HTTP GET（复用共享连接池；单笔超时下调，失败快速退出）。"""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            client = await _http_client()
            return await client.get(url, timeout=timeout, headers=headers or {})
        except Exception as e:
            last_exc = e
            if attempt < retries:
                await asyncio.sleep(base_delay * (2 ** attempt))
    raise last_exc


# 行情源健康状态：上游（新浪/东财）连续降级时记录，供调用方区分
# 「板块列表拉不到（网络故障）」与「板块本身无匹配股票」两类失败，避免误报“样本不足”。
_market_status: dict = {"degraded": False, "last_error": "", "ts": 0.0}


def market_list_health() -> dict:
    """返回最近一次板块列表拉取的健康状态（降级标记 + 错误信息）。"""
    return dict(_market_status)


async def fetch_kline_retry(code: str, days: int, attempts: int = 3,
                            delay: float = 0.3) -> list[dict]:
    """拉日K线带重试：上游限流/超时返回空或抛错时快速重试，
    避免回测/打分高并发场景下单次网络抖动静默丢票。"""
    for attempt in range(attempts):
        try:
            kl = await fetch_kline(code, days)
            if kl:
                return kl
        except Exception:
            pass
        if attempt < attempts - 1:
            await asyncio.sleep(delay * (attempt + 1))
    return []


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
        # 北交所市场位 f13=0 与深市相同，须按代码前缀区分 bj（8/920/4 开头），否则回包键对不上
        code = _eastmoney_code(item.get("f12", ""), item.get("f13", 0))
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
    # 指数/ETF（代码前缀匹配，需配合 fetch_index_list 拉取成分）
    "hs300": lambda c: True,       # 沪深300（需从成分表过滤）
    "zz500": lambda c: True,       # 中证500
    "etf": lambda c: c.startswith(("sh51", "sz159")),
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
MARKET_CACHE_TTL = 300  # 秒（旧版 60s 太短，行情页频繁重复拉取）


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
    resp = await client.get(_SINA_MARKET_URL, params=params, headers=_SINA_HEADERS, timeout=10)
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
    多主机降级（push2 主站可能被网络拦截 → push2delay）。
    """
    fs = _EASTMONEY_FS.get(board, _EASTMONEY_FS["all"])
    matcher = BOARD_MATCHERS.get(board, BOARD_MATCHERS["all"])
    limit = max(1, min(limit, 3000))
    try:
        data = await _eastmoney_clist({
            "pn": 1, "pz": limit, "po": 1, "np": 1, "fltt": 2, "invt": 2,
            "fid": "f6", "fs": fs, "fields": "f12,f13,f14,f2,f3,f5,f6,f8,f9,f23,f20,f21",
        })
        rows = (data or {}).get("data", {}).get("diff") or []
    except Exception:
        return []
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
    board 支持 "sector:<行业名>"（如 "sector:半导体"）：拉取全市场后用
    fetch_sector_map 的 code→行业 映射过滤出目标行业成分池，供选股/回测按行业板块使用。
    """
    if isinstance(board, str) and board.startswith("sector:"):
        sector_name = board.split(":", 1)[1].strip()
        limit = max(1, min(limit, 3000))
        cache_key = f"{board}:{limit}:{sort_field}"
        cached = _market_cache.get(cache_key)
        if cached and time.time() - cached[0] < MARKET_CACHE_TTL:
            return cached[1]
        rows = await fetch_market_list("all", 3000, sort_field)
        if not rows:
            return []
        try:
            sector_map = await fetch_sector_map()
        except Exception:
            sector_map = {}
        out = [r for r in rows if sector_map.get(r["code"]) == sector_name][:limit]
        if out:
            _market_cache[cache_key] = (time.time(), out)
        return out

    # 指数成分板块（hs300/zz500）：按真实成分股过滤，消除“挂名”却不过滤的跨模块不一致
    # （选股 vs 回测/ML 对同一“沪深300”含义不同的历史 bug）
    if board in ("hs300", "zz500"):
        index_code = {"hs300": "sh000300", "zz500": "sh000905"}[board]
        codes = await fetch_index_constituents(index_code)
        if not codes:
            _market_status["degraded"] = True
            _market_status["last_error"] = f"{board}: 成分股列表获取失败（东财）"
            _market_status["ts"] = time.time()
            return []
        rows = await fetch_market_list("all", 3000, sort_field)
        by_code = {r["code"]: r for r in rows}
        out = [by_code[c] for c in codes if c in by_code][:limit]
        if out:
            _market_cache[cache_key] = (time.time(), out)
        return out

    matcher = BOARD_MATCHERS.get(board, BOARD_MATCHERS["all"])
    limit = max(1, min(limit, 3000))
    cache_key = f"{board}:{limit}:{sort_field}"
    cached = _market_cache.get(cache_key)
    if cached and time.time() - cached[0] < MARKET_CACHE_TTL:
        return cached[1]

    try:
        rows = await _fetch_sina_market(board, limit, sort_field, matcher)
        _market_status["degraded"] = False
        _market_status["last_error"] = ""
    except Exception as e:
        _market_status["degraded"] = True
        _market_status["last_error"] = f"sina: {e}"
        _market_status["ts"] = time.time()
        rows = []
    if not rows:
        # 新东财降级：新浪限流/返回空时用东财 clist 兜底，避免“网络一差→股票池全空→全线报样本不足”
        try:
            rows = await fetch_eastmoney_market(board, limit)
            if rows:
                _market_status["degraded"] = False
                _market_status["last_error"] = ""
        except Exception as e:
            if not _market_status["last_error"]:
                _market_status["last_error"] = f"eastmoney: {e}"
            _market_status["degraded"] = True
            _market_status["ts"] = time.time()
            rows = []

    # 空结果禁止写缓存：旧版把空列表也缓存 60s，故障被放大；空时强制下次重拉
    if rows:
        _market_cache[cache_key] = (time.time(), rows)
    return rows


async def fetch_market_list_multi(boards: list[str], limit: int = 300,
                                  sort_field: str = "amount") -> list[dict]:
    """多板块 OR 合并候选池：逐板块取池后去重合并（板块可组合，如「沪市主板+创业板」）。

    hs300/zz500 走真实成分股（fetch_market_list 内部已处理），合并结果按原序去重。
    """
    boards = [b for b in (boards or []) if b] or ["all"]
    out: list[dict] = []
    seen: set[str] = set()
    per = max(10, limit // len(boards))
    for b in boards:
        try:
            rows = await fetch_market_list(b, per, sort_field)
        except Exception:
            rows = []
        for r in rows:
            if r["code"] not in seen:
                seen.add(r["code"])
                out.append(r)
    return out[:limit]


async def _fetch_sina_market(board: str, limit: int, sort_field: str, matcher) -> list[dict]:
    """新浪 Market_Center 分页拉取 + 板块过滤。失败抛异常由上层降级。"""
    matched: list[dict] = []
    seen_codes: set[str] = set()
    client = await _http_client()
    page = 1
    batch_size = 10  # 每批并发页数（旧版 5，翻页更慢）
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
    bj 代码中部分 8xxxxx（如 830799）腾讯 fqkline 返回空 day，自动降级到东财日K。
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

    # 北交所：腾讯 8xxxxx 空 day、920xxx 仅当日一根 → 先降级新浪日K（920xxx 全历史），
    # 仍空再东财日K兜底（东财 secid=0.xxxx 对 920xxx 无数据、对 8xxxxx 有旧历史）
    if code.startswith("bj") and len(kline) < min(20, days):
        kline = await _fetch_sina_kline(code, days)
        if not kline:
            kline = await _fetch_eastmoney_kline(code, days)

    # 空结果禁止写缓存：旧版把限流返回的空 K 线也落盘+入内存缓存，此后 300s 内
    # 所有调用都拿到空数据，单次网络抖动被放大成 5 分钟故障。
    if not kline:
        return []
    db.upsert_kline(code, kline)
    merged = db.get_cached_kline(code)
    result = merged[-days:]
    _kline_cache_set(cache_key, (time.time(), result))
    return result


async def _fetch_sina_kline(code: str, days: int) -> list[dict]:
    """新浪日K线（北交所主力源：920xxx 全历史，腾讯仅当日；8xxxxx 数据止于换码前）。"""
    url = ("https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"
           f"?symbol={code}&scale=240&ma=no&datalen={max(1, days)}")
    try:
        resp = await _http_get(url, 10)
        rows = json.loads(resp.text) if resp.text.strip().startswith("[") else []
    except Exception:
        return []
    out = []
    for r in rows:
        try:
            out.append({
                "date": r["day"],
                "open": float(r["open"]), "close": float(r["close"]),
                "high": float(r["high"]), "low": float(r["low"]),
                "volume": float(r["volume"]),
            })
        except (TypeError, ValueError, KeyError):
            continue
    return out


async def _fetch_eastmoney_kline(code: str, days: int) -> list[dict]:
    """东财日K线（北交所兜底，前复权）。返回与 fetch_kline 同构的列表。"""
    market = "1" if code.startswith("sh") else "0"
    pure = code[2:]
    url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get"
           f"?secid={market}.{pure}&fields1=f1,f2,f3"
           "&fields2=f51,f52,f53,f54,f55,f56,f57"
           f"&klt=101&fqt=1&end=20500101&lmt={max(1, days)}")
    try:
        resp = await _http_get(url, 10)
        klines = (resp.json() or {}).get("data", {}).get("klines") or []
    except Exception:
        return []
    out = []
    for line in klines:
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            out.append({
                "date": parts[0],
                "open": float(parts[1]), "close": float(parts[2]),
                "high": float(parts[3]), "low": float(parts[4]),
                "volume": float(parts[5]),
            })
        except (TypeError, ValueError):
            continue
    return out


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
    """分钟级 K 线（新浪）。

    period: "1"=1分钟, "5"=5分钟, "15"=15分钟, "30"=30分钟, "60"=60分钟。
    count: K 线数量。默认 240（≈ 一个交易日的 1 分钟数）。
    返回 [{datetime, open, close, high, low, volume}]，按时间升序。

    注：原腾讯 web.ifzq.gtimg.cn 分钟接口已 301 跳 web3 域名，部分网络 DNS 解析失败；
    新浪 money.finance.sina.com.cn 接口稳定且直接返回 JSON。
    """
    cache_key = f"{code}:{period}:{count}"
    cached = _minute_cache.get(cache_key)
    if cached and time.time() - cached[0] < MINUTE_CACHE_TTL:
        return cached[1]

    # 新浪 scale 与 period 一致（1/5/15/30/60）
    url = (f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol={code}&scale={period}&datalen={count}")
    resp = await _http_get(url, 10, headers=_SINA_HEADERS)
    arr = resp.json() or []
    out = []
    for row in arr:
        try:
            out.append({
                "datetime": row["day"],
                "open": float(row["open"]), "close": float(row["close"]),
                "high": float(row["high"]), "low": float(row["low"]),
                "volume": float(row["volume"]),
            })
        except (KeyError, ValueError, TypeError):
            continue
    _minute_cache[cache_key] = (time.time(), out)
    return out


# ---------------- 东方财富 clist 多主机降级 ----------------

_EASTMONEY_CLIST_HOSTS = ("push2.eastmoney.com", "push2delay.eastmoney.com")
_EASTMONEY_HOST_OK: str | None = None  # 记录最近可用主机，避免每次分页都重试挂掉的主机


async def _eastmoney_clist(params: dict) -> dict:
    """调用东财 clist 接口，主/备用主机依次尝试（push2 主站可能被网络拦截）。"""
    global _EASTMONEY_HOST_OK
    last_err: Exception | None = None
    hosts = list(_EASTMONEY_CLIST_HOSTS)
    if _EASTMONEY_HOST_OK in hosts:
        hosts.remove(_EASTMONEY_HOST_OK)
        hosts.insert(0, _EASTMONEY_HOST_OK)
    for host in hosts:
        try:
            url = f"https://{host}/api/qt/clist/get?{urlencode(params)}"
            resp = await _http_get(url, 10)
            _EASTMONEY_HOST_OK = host
            return resp.json() or {}
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    return {}


async def _eastmoney_clist_pages(fs: str, fields: str, fid: str = "f3",
                                 page_size: int = 100, max_pages: int = 60) -> list[dict]:
    """分页拉取东财 clist（pz 上限 100，需翻页）。返回去重后的原始行列表。"""
    async def fetch_page(pn: int) -> list:
        try:
            data = await _eastmoney_clist({
                "pn": pn, "pz": page_size, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                "fid": fid, "fs": fs, "fields": fields,
            })
            rows = (data.get("data") or {}).get("diff") or []
            return rows.values() if isinstance(rows, dict) else rows
        except Exception:
            return []

    pages: list[list] = [await fetch_page(1)]
    if not pages[0]:
        return []
    pn = 2
    while pn <= max_pages:
        end = min(pn + 8, max_pages + 1)
        batch = await asyncio.gather(*(fetch_page(p) for p in range(pn, end)))
        got_any = False
        for rows in batch:
            if rows:
                got_any = True
            pages.append(rows)
        pn = end
        if not got_any:
            break
    seen: set[str] = set()
    out = []
    for rows in pages:
        for item in rows:
            key = str(item.get("f12", ""))
            if key and key not in seen:
                seen.add(key)
                out.append(item)
    return out


# ---------------- 指数成分股（东方财富） ----------------

_index_constituent_cache: dict[str, tuple[float, list[str]]] = {}
INDEX_CACHE_TTL = 3600  # 1h，成分股变动低频

# 指数 → 东财板块代码（fs=b:BKxxxx 拉成分，push2 实测有效）
_INDEX_BOARD_MAP = {
    "sh000300": "BK0500",   # 沪深300
    "sh000905": "BK0701",   # 中证500
}


async def fetch_index_constituents(index_code: str) -> list[str]:
    """拉取指数成分股列表（沪深300/中证500），返回 sh/sz 前缀代码列表。"""
    cache_key = index_code
    cached = _index_constituent_cache.get(cache_key)
    if cached and time.time() - cached[0] < INDEX_CACHE_TTL:
        return cached[1]

    board = _INDEX_BOARD_MAP.get(index_code)
    if not board:
        return []

    try:
        rows = await _eastmoney_clist_pages(f"b:{board}", "f12,f13", max_pages=6)
        codes = []
        for item in rows:
            code = _eastmoney_code(item.get("f12", ""), item.get("f13", 0))
            if code:
                codes.append(code)
        _index_constituent_cache[cache_key] = (time.time(), codes)
        return codes
    except Exception:
        return _index_constituent_cache.get(cache_key, (0, []))[1]


# ---------------- 行业分类（东方财富） ----------------

_sector_cache: dict[str, tuple[float, dict[str, str]]] = {}
SECTOR_CACHE_TTL = 86400  # 24h，行业分类变动极低频


async def fetch_sector_map() -> dict[str, str]:
    """拉取全市场股票→行业映射（code → sector_name）。

    返回 {sh600519: "食品饮料", ...}，用于行业因子/中性化。
    """
    cache_key = "sector_map"
    cached = _sector_cache.get(cache_key)
    if cached and time.time() - cached[0] < SECTOR_CACHE_TTL:
        return cached[1]

    try:
        rows = await _eastmoney_clist_pages(
            "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
            "f12,f13,f100", max_pages=70)
        sector_map = {}
        for item in rows:
            code = _eastmoney_code(item.get("f12", ""), item.get("f13", 0))
            sector = item.get("f100", "")
            if code and sector:
                sector_map[code] = sector
        _sector_cache[cache_key] = (time.time(), sector_map)
        return sector_map
    except Exception:
        return _sector_cache.get(cache_key, (0, {}))[1]


# ---------------- 宏观数据（东方财富数据中心） ----------------

# 指标 → (报表名, 取值字段, 单位)。字段名经真实响应核对：
# CPI/PPI 用"同比"，PMI 用制造业指数，M2 用货币供应量同比。
_MACRO_REPORTS = {
    "CPI": ("RPT_ECONOMY_CPI", "NATIONAL_SAME", "同比%"),
    "PPI": ("RPT_ECONOMY_PPI", "BASE_SAME", "同比%"),
    "PMI": ("RPT_ECONOMY_PMI", "MAKE_INDEX", "指数"),
    "M2": ("RPT_ECONOMY_CURRENCY_SUPPLY", "BASIC_CURRENCY_SAME", "同比%"),
}


async def fetch_macro_indicator(indicator: str) -> dict | None:
    """拉取宏观指标（CPI/PPI/PMI/M2），返回最新一期值。"""
    spec = _MACRO_REPORTS.get((indicator or "").strip().upper())
    if not spec:
        return None
    report, value_field, unit = spec
    url = (f"https://datacenter-web.eastmoney.com/api/data/v1/get"
           f"?reportName={report}&columns=ALL&pageSize=1&sortColumns=REPORT_DATE&sortTypes=-1")
    try:
        resp = await _http_get(url, 10)
        records = (resp.json() or {}).get("result", {}).get("data") or []
        if not records:
            return None
        r = records[0]
        return {
            "indicator": (indicator or "").strip().upper(),
            "value": r.get(value_field),
            "date": r.get("REPORT_DATE") or r.get("REGULAR_DATE"),
            "unit": unit,
        }
    except Exception:
        return None


# ---------------- 期货/期权行情（中金所→新浪 hq；商品→东财 ulist，双源保新鲜） ----------------

_future_cache: dict[str, tuple[float, dict]] = {}
FUTURE_CACHE_TTL = 30

# 中金所品种（新浪 CFF_ 前缀实时新鲜）
_CFF_FAMILIES = {"IF", "IH", "IC", "IM", "T", "TF", "TS"}

# 商品期货品种 → 东财市场代码（113上期/114大商/115郑商/143广期）。
# 新浪 hq.sinajs.cn 商品连续合约（RB0/AU0）实测返回 2024 年陈旧快照，改用东财 ulist。
_FUTURE_MARKET: dict[str, int] = {}
for _f in ("CU", "AL", "ZN", "PB", "NI", "SN", "AU", "AG", "RB", "HC", "SS", "WR",
           "FU", "BU", "RU", "SP", "AO", "BR", "EC", "LU", "NR", "SC"):
    _FUTURE_MARKET[_f] = 113
for _f in ("A", "B", "C", "CS", "M", "Y", "P", "J", "JM", "I", "L", "V", "PP",
           "EG", "PG", "RR", "LH", "EB", "JD", "FB", "BB"):
    _FUTURE_MARKET[_f] = 114
for _f in ("SR", "CF", "TA", "OI", "MA", "FG", "RM", "ZC", "SF", "SM", "AP", "CJ",
           "CY", "UR", "SA", "PF", "PK", "PX", "SH", "WH", "PM", "RI", "LR", "JR", "RS", "RO"):
    _FUTURE_MARKET[_f] = 115
_FUTURE_MARKET["SI"] = 143
_FUTURE_MARKET["LC"] = 143

# 期货主力连续合约池（中金所 IF/IH/IC/IM/T/TF/TS + 商品主连 XX0），
# 供 ML 训练/回测按 assetClass=future 取池。新浪日K接口支持 XX0 主连写法。
FUTURE_UNIVERSE = [
    "IF0", "IH0", "IC0", "IM0", "T0", "TF0", "TS0",
    "CU0", "AL0", "ZN0", "PB0", "NI0", "SN0", "AU0", "AG0", "RB0", "HC0", "SS0",
    "WR0", "FU0", "BU0", "RU0", "SP0", "AO0", "BR0", "EC0", "LU0", "NR0", "SC0",
    "A0", "B0", "C0", "CS0", "M0", "Y0", "P0", "J0", "JM0", "I0", "L0", "V0", "PP0",
    "EG0", "PG0", "RR0", "LH0", "EB0", "JD0", "FB0", "BB0",
    "SR0", "CF0", "TA0", "OI0", "MA0", "FG0", "RM0", "ZC0", "SF0", "SM0", "AP0", "CJ0",
    "CY0", "UR0", "SA0", "PF0", "PK0", "PX0", "SH0", "WH0", "PM0", "RI0", "LR0", "JR0",
    "RS0", "RO0", "SI0", "LC0",
]


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _future_secid(code: str) -> str:
    """新浪期货代码（仅中金所使用）：IF2608 → CFF_IF2608。"""
    c = code.strip().upper()
    for prefix in ("CFF_", "NF_"):
        if c.startswith(prefix):
            c = c[len(prefix):]
    family = c.rstrip("0123456789")
    if family in _CFF_FAMILIES:
        return f"CFF_{c}"
    return f"{family[0]}{family[1:].lower()}0".upper() if family else c


def _future_em_symbol(code: str) -> str | None:
    """商品期货 → 东财 secid（如 113.au2612 / 113.rbm 主连）。

    纯品种（au）或 X0（AU0）连续写法 → 东财主连 {family}m；带月份（au2612）原样。
    """
    c = (code or "").strip()
    if not c:
        return None
    up = c.upper()
    for prefix in ("CFF_", "NF_"):
        if up.startswith(prefix):
            up = up[len(prefix):]
            c = c[len(prefix):]
    family = up.rstrip("0123456789")
    market = _FUTURE_MARKET.get(family)
    if not market:
        return None
    if family == up or up.endswith("0"):
        symbol = f"{family.lower()}m"          # 主连：rbm/aum/mm
    else:
        symbol = c if market == 115 else c.lower()   # 郑商所代码保留大写
    return f"{market}.{symbol}"


def _parse_sina_future(text: str) -> dict[str, dict]:
    """解析新浪 hq.sinajs.cn 期货响应（中金所格式，parts[0]=最新价）。"""
    out: dict[str, dict] = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("var hq_str_") or "=" not in line:
            continue
        key, raw = line.split("=", 1)
        secid = key.replace("var hq_str_", "").strip()
        raw = raw.strip().strip(";").strip('"')
        parts = raw.split(",")
        if len(parts) < 15 or not parts[0] or not _is_number(parts[0]):
            continue

        def f(idx):
            try:
                return float(parts[idx])
            except (IndexError, ValueError):
                return 0.0

        price = f(0)
        if price <= 0:
            continue
        pre_settle = f(3) or f(1)
        open_interest = f(14) if f(14) > 0 else f(12)
        out[secid] = {
            "name": secid, "code": secid,
            "price": price, "preSettle": pre_settle, "preClose": pre_settle,
            "open": f(2), "high": price, "low": price,
            "volume": f(4), "amount": f(5), "openInterest": open_interest,
            "changePct": (price / pre_settle - 1.0) if pre_settle > 0 else 0.0,
        }
    return out


async def fetch_future_quotes(codes: list[str]) -> dict:
    """期货实时行情。

    - 中金所（IF/IH/IC/IM/T/TF/TS）→ 新浪 hq.sinajs.cn（实时新鲜，已验证）
    - 商品期货 → 东财 push2delay ulist（fltt=2 自动按合约精度格式化，避免新浪陈旧快照）
    codes 示例: ["IF2608", "rb2610", "au2612", "AU0"]
    返回 {原代码: {name, price, preClose, open, high, low, volume, amount, openInterest, changePct}}
    """
    if not codes:
        return {}
    cache_key = ",".join(codes)
    cached = _future_cache.get(cache_key)
    if cached and time.time() - cached[0] < FUTURE_CACHE_TTL:
        return cached[1]

    out: dict[str, dict] = {}
    sina_codes, em_codes = [], []
    for c in codes:
        up = c.strip().upper()
        for prefix in ("CFF_", "NF_"):
            if up.startswith(prefix):
                up = up[len(prefix):]
        if up.rstrip("0123456789") in _CFF_FAMILIES:
            sina_codes.append(c)
        else:
            em_codes.append(c)

    # 中金所 → 新浪
    if sina_codes:
        secids = [_future_secid(c) for c in sina_codes]
        try:
            resp = await _http_get(f"https://hq.sinajs.cn/list={','.join(secids)}",
                                   10, headers=_SINA_HEADERS)
            parsed = _parse_sina_future(resp.content.decode("gbk", errors="ignore"))
            for c in sina_codes:
                secid = _future_secid(c)
                if secid in parsed:
                    out[c] = parsed[secid]
        except Exception:
            pass

    # 商品 → 东财 ulist
    if em_codes:
        em_secids: list[str] = []
        em2orig: dict[str, str] = {}
        for c in em_codes:
            s = _future_em_symbol(c)
            if s:
                em_secids.append(s)
                # 响应按 f12（symbol，无市场前缀）回填原始代码
                em2orig.setdefault(s.split(".", 1)[1], c)
        if em_secids:
            url = ("https://push2delay.eastmoney.com/api/qt/ulist.np/get"
                   f"?fltt=2&invt=2&secids={','.join(em_secids)}"
                   "&fields=f2,f3,f4,f5,f6,f12,f14,f15,f16,f17,f18")
            try:
                resp = await _http_get(url, 10)
                rows = (resp.json() or {}).get("data", {}).get("diff") or []
                for item in (rows.values() if isinstance(rows, dict) else rows):
                    orig = em2orig.get(item.get("f12", ""))
                    if not orig:
                        continue
                    price = _num(item, "f2")
                    if price is None:
                        continue
                    out[orig] = {
                        "name": item.get("f14", orig), "code": orig,
                        "price": price,
                        "preSettle": _num(item, "f18") or 0.0,
                        "preClose": _num(item, "f18") or 0.0,
                        "open": _num(item, "f17") or 0.0,
                        "high": _num(item, "f15") or 0.0,
                        "low": _num(item, "f16") or 0.0,
                        "volume": _num(item, "f5") or 0.0,
                        "amount": _num(item, "f6") or 0.0,
                        "openInterest": 0.0,   # ulist 无持仓量字段，K线接口可提供
                        "changePct": _num(item, "f3") or 0.0,
                    }
            except Exception:
                pass

    _future_cache[cache_key] = (time.time(), out)
    return out


async def fetch_future_kline(code: str, days: int = 150) -> list[dict]:
    """期货日K线（新浪 InnerFuturesNewService）。

    code 示例: "IF2608"（无需交易所前缀）
    返回 [{date, open, close, high, low, volume, openInterest}]
    """
    symbol = code.strip().upper()
    for prefix in ("CFF_", "NF_"):
        if symbol.startswith(prefix):
            symbol = symbol[len(prefix):]
            break
    url = (f"https://stock2.finance.sina.com.cn/futures/api/jsonp.php/"
           f"var%20t=/InnerFuturesNewService.getDailyKLine?symbol={symbol}")
    try:
        resp = await _http_get(url, 10, headers=_SINA_HEADERS)
        text = resp.text
        if "(" not in text or not text.endswith(");"):
            return []
        payload = text[text.index("(") + 1: text.rindex(")")]
        rows = json.loads(payload)
        out = []
        for r in rows[-days:]:
            try:
                out.append({
                    "date": r["d"],
                    "open": float(r["o"]), "high": float(r["h"]),
                    "low": float(r["l"]), "close": float(r["c"]),
                    "volume": float(r["v"]),
                    "openInterest": float(r.get("p") or 0),
                })
            except (TypeError, ValueError, KeyError):
                continue
        return out
    except Exception:
        return []


# ---------------- 分时图 ----------------

_TIME_SHARE_CACHE: dict[str, tuple[float, list]] = {}
_TIME_SHARE_TTL = 60  # 秒（盘中高频刷新）


async def fetch_time_share(code: str) -> list[dict]:
    """拉取当日分时数据（1分钟价格+成交量+均价黄线）。

    优先用新浪 minute K 线 per="1", count=240，当日盘中数据。
    返回 [{time, price, volume, avgPrice}]，按时间升序。无数据时返回 []。
    """
    cache_key = code
    cached = _TIME_SHARE_CACHE.get(cache_key)
    if cached and time.time() - cached[0] < _TIME_SHARE_TTL:
        return cached[1]

    try:
        url = (f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
               f"CN_MarketData.getKLineData?symbol={code}&scale=1&datalen=240")
        resp = await _http_get(url, 10, headers=_SINA_HEADERS)
        arr = resp.json() or []
    except Exception:
        arr = []

    out = []
    cum_amount = 0.0
    cum_vol = 0.0
    for row in arr:
        try:
            prc = float(row["close"])
            vol = float(row["volume"])
            cum_amount += prc * vol
            cum_vol += vol
            avg = cum_amount / cum_vol if cum_vol > 0 else prc
            out.append({
                "time": row["day"].split(" ")[-1][:5],
                "price": prc,
                "volume": vol,
                "avgPrice": round(avg, 2),
            })
        except (KeyError, ValueError, TypeError):
            continue

    _TIME_SHARE_CACHE[cache_key] = (time.time(), out)
    return out

async def get_sector_exposures(codes: list[str]) -> list[list[float]]:
    """对一组股票代码返回行业哑变量矩阵，供选股中性化使用。"""
    from .factors import sector_dummies
    sector_map = await fetch_sector_map()
    return sector_dummies(codes, sector_map)