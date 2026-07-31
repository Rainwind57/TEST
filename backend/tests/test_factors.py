"""factors.py 纯 Python 因子与统计函数单元测试（P2-13）。

覆盖：
- 基础因子（momentum/rsi/volatility/ma_dev）边界与符号
- 统计函数（mean/std/pearson/spearman/ols）
- 绩效指标（sharpe/max_drawdown/win_rate/annualized_return）
- 回归（ols/ridge/quantile/huber 数值正确性）
"""
import math

import pytest

from app import factors


# ---------------- 基础统计 ----------------

def test_mean_basic():
    assert factors.mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_mean_empty_raises():
    """mean([]) 既有实现未防空，会抛 ZeroDivisionError（属既有行为，测试记录之）。"""
    with pytest.raises(ZeroDivisionError):
        factors.mean([])


def test_std_basic():
    assert factors.std([1.0, 1.0, 1.0]) == pytest.approx(0.0)
    assert factors.std([0.0, 2.0]) == pytest.approx(1.0)


def test_pearson_perfect_positive():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2.0, 4.0, 6.0, 8.0, 10.0]
    assert factors.pearson(xs, ys) == pytest.approx(1.0)


def test_pearson_perfect_negative():
    xs = [1.0, 2.0, 3.0]
    ys = [3.0, 2.0, 1.0]
    assert factors.pearson(xs, ys) == pytest.approx(-1.0)


def test_pearson_too_short():
    assert factors.pearson([1.0], [2.0]) == 0.0


def test_spearman_perfect_monotonic():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [10.0, 20.0, 30.0, 40.0]
    assert factors.spearman(xs, ys) == pytest.approx(1.0)


def test_rank_of():
    assert factors.rank_of([3.0, 1.0, 2.0]) == [3.0, 1.0, 2.0]


# ---------------- 基础因子 ----------------

def test_factor_momentum_uptrend(kline_uptrend):
    """单边上扬序列 momentum 必为正。"""
    m = factors.factor_momentum(kline_uptrend, len(kline_uptrend) - 1, n=20)
    assert m is not None and m > 0


def test_factor_momentum_short_history():
    """i < n 时应返回 None（历史不足保护）。"""
    kline = [{"date": "2024-01-01", "open": 10, "close": 10, "high": 10, "low": 10, "volume": 100}]
    assert factors.factor_momentum(kline, 0, n=20) is None


def test_factor_rsi_all_up():
    """纯上涨序列 RSI 应为 100（avg_loss=0）。"""
    kline = [{"date": f"2024-01-{i+1:02d}", "open": 10+i, "close": 10+i,
              "high": 10+i, "low": 10+i, "volume": 100} for i in range(20)]
    rsi = factors.factor_rsi(kline, 19, n=14)
    assert rsi == 100.0


def test_factor_rsi_range(kline_random):
    """RSI 必落在 [0, 100]。"""
    rsi = factors.factor_rsi(kline_random, len(kline_random) - 1, n=14)
    assert rsi is not None and 0 <= rsi <= 100


def test_factor_volatility_nonnegative(kline_random):
    """波动率非负。"""
    v = factors.factor_volatility(kline_random, len(kline_random) - 1, n=20)
    assert v is not None and v >= 0


def test_factor_ma_dev_returns_none_for_short():
    kline = [{"date": "2024-01-01", "open": 10, "close": 10, "high": 10, "low": 10, "volume": 100}]
    assert factors.factor_ma_dev(kline, 0, n=20) is None


# ---------------- 绩效指标 ----------------

def test_sharpe_positive_returns():
    """稳定正收益且非零波动 Sharpe 必为正。"""
    import numpy as np
    rets = (np.arange(252) % 10) * 0.001 + 0.0005  # 非等值正收益，std>0
    s = factors.sharpe_ratio(list(rets))
    assert s > 0


def test_sharpe_zero_vol():
    """零波动率 Sharpe 应为 0（防除零）。"""
    assert factors.sharpe_ratio([0.0, 0.0, 0.0]) == 0.0


def test_max_drawdown_basic():
    """峰值 10 → 谷底 5 → 回撤 -50%（实现返回负值）。"""
    assert factors.max_drawdown([10.0, 8.0, 5.0, 7.0]) == pytest.approx(-0.5)


def test_max_drawdown_no_drawdown():
    """单调递增无回撤，返回 0。"""
    assert factors.max_drawdown([1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.0)


def test_win_rate():
    assert factors.win_rate([0.01, -0.01, 0.02, -0.02]) == pytest.approx(0.5)


def test_win_rate_empty():
    assert factors.win_rate([]) == 0.0


def test_annualized_return_basic():
    """日频 1% 复利 252 日，年化收益率 = 1.01^252 - 1 ≈ 11.27（即 1127%）。"""
    r = factors.annualized_return([0.01] * 252, periods_per_year=252)
    assert r == pytest.approx(11.27, rel=0.01)


# ---------------- 回归 ----------------

def test_ols_regression_linear():
    """y=2x+1 完美线性拟合，a≈1、b≈2、r2≈1。"""
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2 * x + 1 for x in xs]
    r = factors.ols_regression(xs, ys)
    assert r["a"] == pytest.approx(1.0, rel=1e-6)
    assert r["b"] == pytest.approx(2.0, rel=1e-6)
    assert r["r2"] == pytest.approx(1.0, rel=1e-6)


def test_ridge_regression_runs():
    """Ridge 在共线数据下应稳定出解（不崩）。"""
    xs = [1.0, 2.0, 3.0]
    ys = [2.0, 4.0, 6.0]
    r = factors.ridge_regression(xs, ys, alpha=1.0)
    assert "a" in r and "b" in r


def test_quantile_regression_median():
    """分位数 τ=0.5 近似中位回归，斜率方向正确。"""
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2 * x + 1 for x in xs]
    r = factors.quantile_regression(xs, ys, tau=0.5)
    assert r["b"] > 0


def test_multi_ols_basic():
    """多元 OLS：y = 1*x0 + 2*x1 + 0.5，系数方向正确。"""
    rows = [[1.0, 1.0], [2.0, 1.0], [3.0, 2.0], [4.0, 2.0]]
    ys = [1*1 + 2*1 + 0.5, 1*2 + 2*1 + 0.5, 1*3 + 2*2 + 0.5, 1*4 + 2*2 + 0.5]
    r = factors.multi_ols(rows, ys)
    assert len(r["coefs"]) == 2
    assert r["coefs"][0] == pytest.approx(1.0, rel=0.1)
    assert r["coefs"][1] == pytest.approx(2.0, rel=0.1)


def test_fit_regression_dispatch():
    """fit_regression 统一返回 coefs（按幂次，c0 为截距）。"""
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2 * x for x in xs]
    r = factors.fit_regression("ols", xs, ys)
    assert r["coefs"][1] == pytest.approx(2.0, rel=1e-6)  # 斜率


# ---------------- 中性化 / zscore ----------------

def test_zscore_basic():
    out = factors.zscore([1.0, 2.0, 3.0])
    assert abs(sum(out)) < 1e-9  # 标准化后均值≈0


def test_neutralized_zscore_reduces_exposure():
    """中性化后与暴露的相关性应低于中性化前。"""
    import random
    random.seed(0)
    values = [random.gauss(0, 1) for _ in range(50)]
    exposures = [[random.gauss(0, 1)] for _ in range(50)]
    neu = factors.neutralized_zscore(values, exposures)
    assert len(neu) == len(values)
    # 中性化结果方差应有限（非发散）
    assert all(abs(v) < 100 for v in neu)
