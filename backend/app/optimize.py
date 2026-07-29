"""参数寻优模块：Optuna 贝叶斯搜索回测参数。

Walk-Forward 防过拟合：前半样本做参数搜索、后半样本做 OOS 验证，
最终回报 IS/OOS 两段指标，禁止全样本调参后宣称高收益。
结果可回写 saved_strategies 表（2.1）。
"""
import asyncio
import optuna

from . import db
from .routers import selection as sel


def _split_hist(hist: int) -> tuple[int, int]:
    """把历史区间分两半：IS（调参）+ OOS（验证）。返回 (is_hist, oos_hist)。"""
    half = max(60, hist // 2)
    return half, hist - half


def _objective(trial: optuna.Trial, base: dict, is_hist: int) -> float:
    """单次试验：用 IS 区间跑回测，返回 Sharpe（越大越好）。"""
    cfg = {
        "board": base["board"],
        "poolSize": trial.suggest_int("poolSize", 30, 150, step=10),
        "factor": base["factor"],
        "groups": trial.suggest_int("groups", 3, 8),
        "n": trial.suggest_int("n", 1, 10),
        "hist": is_hist,
        "commissionRate": base["commissionRate"],
        "stampDuty": base["stampDuty"],
        "slippage": base["slippage"],
        "benchmark": base["benchmark"],
        "applyCost": base["applyCost"],
    }
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
    返回 {best_params, is_metrics, oos_metrics, trials}。
    """
    is_hist, oos_hist = _split_hist(base_config["hist"])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    for i in range(n_trials):
        if progress_cb:
            progress_cb(i + 1, n_trials)
        study.optimize(lambda t: _objective(t, base_config, is_hist), n_trials=1, catch=(Exception,))

    best = study.best_params if study.best_trial else {}
    best_params = {
        "poolSize": best.get("poolSize", base_config["poolSize"]),
        "groups": best.get("groups", base_config["groups"]),
        "n": best.get("n", base_config["n"]),
    }

    is_metrics = _run_with(base_config, is_hist, best_params)
    oos_metrics = _run_with(base_config, oos_hist, best_params)

    trials = [
        {"number": t.number, "value": t.value, "params": t.params}
        for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    return {
        "bestParams": best_params,
        "isMetrics": is_metrics,
        "oosMetrics": oos_metrics,
        "nTrials": len(trials),
        "trials": trials,
    }


def _run_with(base: dict, hist: int, params: dict) -> dict:
    cfg = {
        "board": base["board"], "factor": base["factor"],
        "poolSize": params.get("poolSize", base["poolSize"]),
        "groups": params.get("groups", base["groups"]),
        "n": params.get("n", base["n"]),
        "hist": hist,
        "commissionRate": base["commissionRate"], "stampDuty": base["stampDuty"],
        "slippage": base["slippage"], "benchmark": base["benchmark"],
        "applyCost": base["applyCost"],
    }
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
    """把最优参数回写为策略（kind=backtest）。"""
    cfg = {
        "board": base_config["board"], "factor": base_config["factor"],
        "poolSize": best_params["poolSize"], "groups": best_params["groups"],
        "n": best_params["n"], "hist": base_config["hist"],
        "commissionRate": base_config["commissionRate"], "stampDuty": base_config["stampDuty"],
        "slippage": base_config["slippage"], "benchmark": base_config["benchmark"],
        "applyCost": base_config["applyCost"],
    }
    return db.create_strategy(name, "backtest", cfg)
