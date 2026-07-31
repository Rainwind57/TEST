"""参数寻优模块：Optuna 贝叶斯搜索回测参数。

Walk-Forward 防过拟合：K 线一次性取足全长，按日期中点切两段互不重叠的窗口——
IS（早期段）做参数搜索、OOS（晚期段）做样本外验证，最终回报 IS/OOS 两段指标，
禁止全样本调参后宣称高收益。结果可回写 saved_strategies 表。

旧版 _split_hist(hist=180) 返回 (90,90)，IS/OOS 都取「最近 90 天」K 线，
两段窗口完全重叠 → OOS 指标 = IS 指标，所谓防过拟合不成立。
"""
import asyncio
import optuna

from . import adapters, db
from .routers import selection as sel


async def _mid_date(board: str, hist: int) -> str | None:
    """拉参考 K 线取日期中点，作为 IS/OOS 不重叠切分点。"""
    try:
        pool = await adapters.fetch_market_list(board, 5)
    except Exception:
        pool = []
    for row in pool:
        try:
            kline = await adapters.fetch_kline(row["code"], hist)
        except Exception:
            continue
        if len(kline) >= 20:
            return kline[len(kline) // 2]["date"]
    return None


def _objective(trial: optuna.Trial, base: dict, full_hist: int, is_end_date: str | None) -> float:
    """单次试验：用 IS 区间（早期段，endDate=中点）跑回测，返回 Sharpe（越大越好）。"""
    cfg = {
        "board": base["board"],
        "poolSize": trial.suggest_int("poolSize", 30, 150, step=10),
        "groups": trial.suggest_int("groups", 3, 8),
        "n": trial.suggest_int("n", 1, 10),
        "hist": full_hist,
        "commissionRate": base["commissionRate"],
        "stampDuty": base["stampDuty"],
        "slippage": base["slippage"],
        "benchmark": base["benchmark"],
        "applyCost": base["applyCost"],
        "endDate": is_end_date,
    }
    # 模型策略与因子策略二选一：有 modelId 走 ML 信号回测，否则技术因子（打通模型寻优）
    if base.get("modelId"):
        cfg["modelId"] = base["modelId"]
    else:
        cfg["factor"] = base["factor"]
    try:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(sel.run_backtest(sel.BacktestBody(**cfg)))
        loop.close()
        return float(result.get("metrics", {}).get("sharpe", 0.0))
    except Exception:
        return -1e9


def optimize_backtest(base_config: dict, n_trials: int = 30, progress_cb=None) -> dict:
    """对回测参数做贝叶斯寻优。

    base_config: 基础配置（board/factor/cost 等固定项）。
    返回 {best_params, is_metrics, oos_metrics, splitDate, trials}。
    """
    full_hist = max(120, base_config["hist"])

    loop = asyncio.new_event_loop()
    try:
        mid_date = loop.run_until_complete(_mid_date(base_config["board"], full_hist))
    finally:
        loop.close()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    for i in range(n_trials):
        if progress_cb:
            progress_cb(i + 1, n_trials)
        study.optimize(lambda t: _objective(t, base_config, full_hist, mid_date),
                       n_trials=1, catch=(Exception,))

    best = study.best_params if study.best_trial else {}
    best_params = {
        "poolSize": best.get("poolSize", base_config["poolSize"]),
        "groups": best.get("groups", base_config["groups"]),
        "n": best.get("n", base_config["n"]),
    }

    is_metrics = _run_with(base_config, full_hist, best_params, end_date=mid_date)
    oos_metrics = _run_with(base_config, full_hist, best_params, start_date=mid_date)

    trials = [
        {"number": t.number, "value": t.value, "params": t.params}
        for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    return {
        "bestParams": best_params,
        "isMetrics": is_metrics,
        "oosMetrics": oos_metrics,
        "splitDate": mid_date,
        "nTrials": len(trials),
        "trials": trials,
    }


def _run_with(base: dict, full_hist: int, params: dict,
              start_date: str | None = None, end_date: str | None = None) -> dict:
    cfg = {
        "board": base["board"],
        "poolSize": params.get("poolSize", base["poolSize"]),
        "groups": params.get("groups", base["groups"]),
        "n": params.get("n", base["n"]),
        "hist": full_hist,
        "commissionRate": base["commissionRate"], "stampDuty": base["stampDuty"],
        "slippage": base["slippage"], "benchmark": base["benchmark"],
        "applyCost": base["applyCost"],
        "startDate": start_date, "endDate": end_date,
    }
    if base.get("modelId"):
        cfg["modelId"] = base["modelId"]
    else:
        cfg["factor"] = base["factor"]
    try:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(sel.run_backtest(sel.BacktestBody(**cfg)))
        loop.close()
        m = result.get("metrics", {})
        return {
            "sharpe": m.get("sharpe"), "annualizedReturn": m.get("annualizedReturn"),
            "maxDrawdown": m.get("maxDrawdown"), "cumulativeReturn": m.get("cumulativeReturn"),
            "winRate": m.get("winRate"), "rebalanceCount": m.get("rebalanceCount"),
        }
    except Exception as e:
        return {"error": str(e)}


def save_best_as_strategy(base_config: dict, best_params: dict, name: str) -> dict:
    """把最优参数回写为策略（kind=backtest）。

    base_config 由前端传入，可能只含部分字段（OptimizeBody 有默认值），用 .get 兜底
    避免缺键 KeyError。默认值与 OptimizeBody 保持一致。
    """
    cfg = {
        "board": base_config.get("board", "all"),
        "poolSize": best_params.get("poolSize", 60),
        "groups": best_params.get("groups", 5),
        "n": best_params.get("n", 5),
        "hist": base_config.get("hist", 180),
        "commissionRate": base_config.get("commissionRate", 0.00025),
        "stampDuty": base_config.get("stampDuty", 0.001),
        "slippage": base_config.get("slippage", 0.001),
        "benchmark": base_config.get("benchmark", "none"),
        "applyCost": base_config.get("applyCost", True),
    }
    if base_config.get("modelId"):
        cfg["modelId"] = base_config["modelId"]
    else:
        cfg["factor"] = base_config.get("factor", "momentum")
    return db.create_strategy(name, "backtest", cfg)
