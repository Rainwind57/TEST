"""定时盯盘调度器：APScheduler 后台任务。

功能：
- 交易日 15:05 自动刷新持仓市值并写 equity_history（日频净值完整）
- 交易日 15:10 重算 watchlist 因子打分，生成三态信号（看多/看空）+ 持仓联动退出建议
- 可选自动调仓：检测到信号后自动生成买卖单（默认关闭，需开启 auto_trade 配置）

默认关闭，需用户显式开启（模拟边界）。内存状态仅作实时返回，关键运行记录
（_last_run / 信号结果）已落 scheduler_runs 表，重启后可审计、可回溯。
"""
import asyncio
import datetime
import os
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from . import db, adapters
from .routers import portfolio as pf
from .factors import FACTORS
from .numpy_factors import kline_to_arrays, compute_factor_series, series_at

_scheduler: AsyncIOScheduler | None = None
_enabled = False
_last_run = None
_last_signals = []

# 交易日历缓存：用上证指数 K 线日期集合推断交易日，替代硬编码 _HOLIDAYS。
# 有 K 线的日期必为交易日，无需维护年度节假日表，也无 2027 失效问题。
_TRADING_DAYS: set[str] | None = None
_TRADING_DAYS_UPDATED: float = 0.0
_TRADING_DAYS_TTL = 24 * 3600  # 一天最多回源一次

# 盯盘信号单股拉取并发上限（取代旧版 codes[:20] 硬截断，全量扫描但限流防刷接口）
_SCAN_CONCURRENCY = 15


def is_enabled() -> bool:
    return _enabled


def get_signal_config() -> dict:
    """读取盯盘信号引擎配置：rule=内置动量/RSI 规则；model=落盘 ML 模型打分。
    
    monitor_ranking 控制模型模式的排名口径：
      - "isolated"（默认）：对 watchlist 各股孤立打分
      - "full"：对全池排名后取 watchlist 子集的分位（与选股 score_latest 口径一致）
    monitor_rule_factor 为规则模式的因子名（空=默认动量+RSI）
    """
    return {
        "mode": db.get_setting("monitor_mode", "rule"),
        "modelId": db.get_setting("monitor_model_id", ""),
        "ranking": db.get_setting("monitor_ranking", "isolated"),
        "ruleFactor": db.get_setting("monitor_rule_factor", ""),
    }


def set_signal_config(mode: str, model_id: str = "", ranking: str = "isolated",
                     rule_factor: str = "") -> dict:
    if mode not in ("rule", "model"):
        raise ValueError("mode 必须为 rule 或 model")
    if mode == "model":
        if not model_id:
            raise ValueError("模型模式下必须指定模型 modelId")
        from . import ml as _ml
        if not os.path.exists(os.path.join(_ml.ML_DIR, f"{model_id}.joblib")):
            raise ValueError(f"模型不存在: {model_id}，请重新选择或删除该配置")
        if ranking not in ("isolated", "full"):
            raise ValueError("ranking 必须为 isolated 或 full")
    db.set_setting("monitor_mode", mode)
    db.set_setting("monitor_model_id", model_id or "")
    db.set_setting("monitor_ranking", ranking if mode == "model" else "")
    db.set_setting("monitor_rule_factor", rule_factor if mode == "rule" else "")
    return get_signal_config()


def last_run():
    return _last_run


def last_signals():
    return _last_signals


def start():
    global _scheduler, _enabled
    if _scheduler:
        return
    _scheduler = AsyncIOScheduler()
    _scheduler.add_job(_refresh_equity, CronTrigger(hour=15, minute=5, day_of_week="mon-fri"))
    _scheduler.add_job(_scan_signals, CronTrigger(hour=15, minute=10, day_of_week="mon-fri"))
    _scheduler.start()
    _enabled = True
    db.set_scheduler_enabled(True)


def stop():
    global _scheduler, _enabled
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
    _enabled = False
    db.set_scheduler_enabled(False)


async def _load_trading_days() -> set[str]:
    """取上证指数 K 线日期集合作为交易日历（带 24h 内存缓存）。

    替代旧版硬编码 _HOLIDAYS（仅 2026 年、未含调休），避免失效及
    调休误判。回源失败时退化为「非周末即交易日」的近似，保证调度不中断。
    """
    global _TRADING_DAYS, _TRADING_DAYS_UPDATED
    import time
    now = time.time()
    if _TRADING_DAYS is not None and now - _TRADING_DAYS_UPDATED < _TRADING_DAYS_TTL:
        return _TRADING_DAYS
    try:
        # 取足够长的历史以覆盖近期所有交易日（含节假日前后）
        kline = await adapters.fetch_kline("sh000001", 400)
        days = {row["date"] for row in kline if row.get("date")}
        if days:
            _TRADING_DAYS = days
            _TRADING_DAYS_UPDATED = now
            return days
    except Exception:
        pass
    # 回源失败：退化近似（非周末即交易日），宁可多跑一次 refresh 也别漏跑
    return _TRADING_DAYS or set()


async def _is_trading_day() -> bool:
    """交易日判断：周末休市 + 交易日历校验（交易日历回源失败时仅判周末）。"""
    now = datetime.datetime.now()
    if now.weekday() >= 5:
        return False
    days = await _load_trading_days()
    if not days:
        return True  # 日历为空（回源失败）时退化为「非周末即交易日」
    return now.strftime("%Y-%m-%d") in days


async def _refresh_equity():
    if not await _is_trading_day():
        return
    global _last_run
    try:
        conn = db.get_conn()
        state = conn.execute("SELECT cash FROM portfolio_state WHERE id = 1").fetchone()
        positions = conn.execute("SELECT code, name, qty, avg_cost FROM positions").fetchall()
        conn.close()
        if not state:
            return
        cash = state["cash"]
        codes = [r["code"] for r in positions] or ["sh000001"]
        quotes = await adapters.fetch_quotes(codes)
        market_value = 0.0
        for p in positions:
            q = quotes.get(p["code"])
            price = q.get("price", 0) if q else 0
            market_value += price * p["qty"]
        total = cash + market_value
        conn = db.get_conn()
        conn.execute("INSERT INTO equity_history (ts, value) VALUES (?, ?)",
                     (datetime.datetime.now().isoformat(), total))
        conn.commit()
        conn.close()
        _last_run = {"task": "refresh_equity", "ts": datetime.datetime.now().isoformat(),
                     "totalValue": total, "cash": cash, "marketValue": market_value}
        db.log_scheduler_run("refresh_equity", True, _last_run)
    except Exception as e:
        _last_run = {"task": "refresh_equity", "error": str(e)}
        db.log_scheduler_run("refresh_equity", False, error=str(e))


async def _scan_signals():
    if not await _is_trading_day():
        return
    await _scan_signals_impl()


async def scan_now(force: bool = False) -> dict:
    """立即手动执行一次盯盘信号扫描（替代只能等交易日 15:10 cron 的被动模式）。

    force=True 时跳过交易日判断，方便用户任意时刻验证扫描逻辑能否产出信号。
    """
    if not force and not await _is_trading_day():
        return {"ok": False, "reason": "当前非交易日，扫描已跳过（可加 force=true 强制扫描）", "signals": []}
    await _scan_signals_impl()
    return {"ok": True, "signals": _last_signals}


async def _scan_signals_impl():
    global _last_signals
    try:
        conn = db.get_conn()
        all_codes = [r["code"] for r in conn.execute("SELECT code FROM watchlist ORDER BY added_at").fetchall()]
        positions = {r["code"]: r for r in conn.execute("SELECT code, name, qty, avg_cost FROM positions").fetchall()}
        conn.close()

        # 过滤不可交易标的（指数/ETF），只对可交易个股生成信号与下单
        codes = [c for c in all_codes if db.is_tradable(c)]
        if not codes:
            _last_signals = []
            return

        cfg = get_signal_config()
        signals: list[dict] = []

        # 模型模式：用落盘 ML 模型预测分生成三态信号 + 持仓联动
        if cfg["mode"] == "model" and cfg.get("modelId"):
            from . import ml
            # 校验模型仍存在：配置残留失效 modelId 时明确报错，而非静默无信号
            if not os.path.exists(os.path.join(ml.ML_DIR, f"{cfg['modelId']}.joblib")):
                _last_signals = []
                db.log_scheduler_run("scan_signals", False,
                                     error=f"模型模式配置的 modelId 已不存在: {cfg['modelId']}，"
                                           f"请重新保存配置（或切回规则模式）")
                return
            try:
                if cfg.get("ranking") == "full":
                    # 全池排名口径（与选股 score_latest 一致）：对全市场排序后取 watchlist 分位
                    ranked = await ml.score_latest(cfg["modelId"], pool_size=150)
                    rank_by_code = {r["code"]: r for r in ranked}
                    total = len(ranked)
                    for c in codes:
                        r = rank_by_code.get(c)
                        if not r:
                            continue
                        pct = (total - r["rank"] + 1) / max(1, total)
                        if pct >= 0.75:
                            sig = "看多"
                        elif pct <= 0.25:
                            sig = "看空"
                        else:
                            sig = "中性"
                        if c in positions and sig == "看空":
                            sig = "平仓"
                        signals.append({"code": c, "score": r["score"], "rank": r["rank"],
                                       "signal": sig, "mode": "model"})
                else:
                    scored = await ml.score_codes(cfg["modelId"], codes)
                    for s in scored:
                        score = s["score"]
                        if score > 0.005:
                            sig = "看多"
                        elif score < -0.005:
                            sig = "看空"
                        else:
                            sig = "中性"
                        if s["code"] in positions and sig == "看空":
                            sig = "平仓"
                        signals.append({
                            "code": s["code"], "score": score, "signal": sig, "mode": "model",
                        })
            except Exception as e:
                _last_signals = []
                db.log_scheduler_run("scan_signals", False, error=f"模型打分失败: {e}")
                return
        else:
            # 规则模式：全量扫描 + 信号量限流 + 三态信号 + 持仓联动
            # 支持自定义因子（monitor_rule_factor）替代默认动量+RSI
            rule_factor = cfg.get("ruleFactor", "")
            sem = asyncio.Semaphore(_SCAN_CONCURRENCY)

            async def one(code: str):
                async with sem:
                    try:
                        kline = await adapters.fetch_kline(code, 60)
                    except Exception:
                        return
                    if len(kline) < 25:
                        return
                    arr = kline_to_arrays(kline)
                    if rule_factor:
                        # 单一因子模式：使用用户指定的因子
                        from .numpy_factors import compute_factor_series
                        import numpy as np
                        fv = compute_factor_series(rule_factor, arr)
                        i = len(kline) - 1
                        v = series_at(fv, i)
                        if v is None:
                            return
                        last_60 = fv[-60:] if len(fv) >= 60 else fv
                        valid = last_60[~np.isnan(last_60)]
                        if len(valid) < 10:
                            return
                        mean60 = float(np.nanmean(valid))
                        std60 = float(np.nanstd(valid)) or 1.0
                        z = (v - mean60) / std60
                        if z > 1.0:
                            tag = f"{rule_factor}强势(看多)"
                        elif z < -1.0:
                            tag = f"{rule_factor}弱势(看空)"
                        else:
                            tag = "中性"
                        if code in positions and "看空" in tag:
                            tag = "减仓"
                        signals.append({"code": code, "factorValue": float(v), "zScore": z,
                                       "signal": tag, "mode": "rule"})
                    else:
                        # 默认双因子模式：动量+RSI
                        mom = compute_factor_series("momentum", arr)
                        rsi = compute_factor_series("rsi", arr)
                        i = len(kline) - 1
                        m = series_at(mom, i)
                        r = series_at(rsi, i)
                        tag = None
                        if r is not None and r < 30:
                            tag = "超卖(看多)"
                        elif r is not None and r > 70:
                            tag = "超买(看空)"
                        if m is not None:
                            if m > 0.1:
                                tag = (tag + "+突破" if tag else "突破(看多)")
                            elif m < -0.05:
                                tag = (tag + "+走弱" if tag else "走弱(看空)")
                        if tag is None:
                            tag = "中性"
                        if code in positions and ("看空" in tag or "走弱" in tag):
                            tag = "减仓"
                        signals.append({"code": code, "momentum": m, "rsi": r, "signal": tag, "mode": "rule"})

            await asyncio.gather(*(one(c) for c in codes))

        # 对已持仓但不在 watchlist 中的标的也做退出检查
        watched = set(codes)
        for pos_code in positions:
            if pos_code in watched:
                continue
            if not db.is_tradable(pos_code):
                continue
            try:
                kline = await adapters.fetch_kline(pos_code, 60)
            except Exception:
                continue
            if len(kline) < 25:
                continue
            arr = kline_to_arrays(kline)
            rsi = compute_factor_series("rsi", arr)
            mom = compute_factor_series("momentum", arr)
            i = len(kline) - 1
            r = series_at(rsi, i)
            m = series_at(mom, i)
            exit_tag = None
            if r is not None and r > 70:
                exit_tag = "持仓超买(建议减仓)"
            if m is not None and m < -0.05:
                exit_tag = ("持仓走弱(建议减仓)"
                           if not exit_tag else exit_tag + "+走弱")
            if exit_tag:
                signals.append({"code": pos_code, "momentum": m, "rsi": r, "signal": exit_tag,
                                "mode": "rule", "inPosition": True})

        _last_signals = signals
        db.log_scheduler_run("scan_signals", True, {"count": len(signals), "mode": cfg["mode"],
                                                     "tradableCount": len(codes)})

        # 自动调仓（需显式开启 auto_trade 配置）
        if db.get_setting("auto_trade", "0") == "1":
            await _execute_auto_trade(signals, positions)
    except Exception as e:
        _last_signals = []
        db.log_scheduler_run("scan_signals", False, error=str(e))


async def _execute_auto_trade(signals: list[dict], positions: dict):
    """根据信号自动生成买卖单（需 auto_trade 配置开启）。

    风险控制：
    - 买单：仅对「看多」信号且未持仓的标的建仓，等权分配，单笔≤总现金 20%
    - 卖单：对「平仓」「减仓」信号的已持仓标的全平
    - 单次买卖各不超过 5 笔
    """
    import datetime as dt

    buy_signals = [s for s in signals if (
        ("看多" in (s.get("signal") or "") or "超卖" in (s.get("signal") or ""))
        and s["code"] not in positions
    )]
    sell_signals = [s for s in signals if (
        s["code"] in positions
        and ("平仓" in (s.get("signal") or "")
             or "减仓" in (s.get("signal") or "")
             or "持仓走弱" in (s.get("signal") or "")
             or "持仓超买" in (s.get("signal") or ""))
    )]

    if not buy_signals and not sell_signals:
        return

    conn = db.get_conn()
    state = conn.execute("SELECT cash FROM portfolio_state WHERE id = 1").fetchone()
    if not state:
        conn.close()
        return
    cash = state["cash"]
    conn.close()

    # 买单：等权分配，最多 5 笔
    max_buy = min(len(buy_signals), 5)
    if max_buy > 0:
        per_buy_cash = min(cash * 0.2, cash / max_buy)
        for sig in buy_signals[:max_buy]:
            try:
                quotes = await adapters.fetch_tencent_quotes([sig["code"]])
            except Exception:
                continue
            q = quotes.get(sig["code"])
            if not q or not q.get("price") or q["price"] <= 0:
                continue
            qty = int(per_buy_cash / q["price"]) // 100 * 100
            if qty < 100:
                continue
            amount = q["price"] * qty
            conn = db.get_conn()
            cur = conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            cur_cash = cur.execute("SELECT cash FROM portfolio_state WHERE id = 1").fetchone()["cash"]
            if amount > cur_cash:
                conn.close()
                continue
            new_cash = cur_cash - amount
            existing = cur.execute("SELECT * FROM positions WHERE code = ?", (sig["code"],)).fetchone()
            if existing:
                new_qty = existing["qty"] + qty
                new_avg = (existing["avg_cost"] * existing["qty"] + amount) / new_qty
                cur.execute("UPDATE positions SET qty=?, avg_cost=?, name=? WHERE code=?",
                           (new_qty, new_avg, q["name"], sig["code"]))
            else:
                cur.execute("INSERT INTO positions (code, name, qty, avg_cost) VALUES (?, ?, ?, ?)",
                           (sig["code"], q["name"], qty, q["price"]))
            cur.execute("UPDATE portfolio_state SET cash=? WHERE id=1", (new_cash,))
            cur.execute("INSERT INTO trades (time, side, code, name, qty, price, amount) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "buy", sig["code"],
                        q["name"], qty, q["price"], amount))
            conn.commit()
            conn.close()

    # 卖单：有卖出信号的持仓全平，最多 5 笔
    max_sell = min(len(sell_signals), 5)
    for sig in sell_signals[:max_sell]:
        pos = positions[sig["code"]]
        try:
            quotes = await adapters.fetch_tencent_quotes([sig["code"]])
        except Exception:
            continue
        q = quotes.get(sig["code"])
        price = q["price"] if q and q.get("price") and q["price"] > 0 else pos["avg_cost"]
        amount = price * pos["qty"]
        conn = db.get_conn()
        cur = conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        cur_pos = cur.execute("SELECT * FROM positions WHERE code=?", (sig["code"],)).fetchone()
        if not cur_pos or cur_pos["qty"] < pos["qty"]:
            conn.close()
            continue
        cur_cash = cur.execute("SELECT cash FROM portfolio_state WHERE id=1").fetchone()["cash"]
        new_cash = cur_cash + amount
        remain = cur_pos["qty"] - pos["qty"]
        if remain <= 0:
            cur.execute("DELETE FROM positions WHERE code=?", (sig["code"],))
        else:
            cur.execute("UPDATE positions SET qty=? WHERE code=?", (remain, sig["code"]))
        cur.execute("UPDATE portfolio_state SET cash=? WHERE id=1", (new_cash,))
        cur.execute("INSERT INTO trades (time, side, code, name, qty, price, amount) VALUES (?,?,?,?,?,?,?)",
                   (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "sell", sig["code"],
                    pos.get("name", sig["code"]), pos["qty"], price, amount))
        conn.commit()
        conn.close()

    db.log_scheduler_run("auto_trade", True, {"buyCount": max_buy, "sellCount": max_sell})
