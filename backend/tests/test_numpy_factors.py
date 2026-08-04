"""numpy_factors.py 向量化因子 + 与 factors.py 同因子数值一致性测试（P2-13）。

两套实现长期需保持数值一致；任何重构若无一致性保障极易引入隐蔽 bug。
逐时间点、逐因子对比，允许浮点误差 1e-9。
"""
import numpy as np
import pytest

from app import factors, numpy_factors as nf


def test_kline_to_arrays_basic(kline_random):
    arr = nf.kline_to_arrays(kline_random)
    assert set(arr.keys()) >= {"date", "close", "high", "low", "volume"}
    assert len(arr["close"]) == len(kline_random)


def test_kline_to_arrays_missing_fields():
    """缺字段会 KeyError（既有行为，无兜底）。"""
    with pytest.raises(KeyError):
        nf.kline_to_arrays([{"date": "2024-01-01", "close": 10}])


def test_series_at_bounds():
    arr = np.array([1.0, 2.0, 3.0])
    assert nf.series_at(arr, 0) == 1.0
    assert nf.series_at(arr, 2) == 3.0
    assert nf.series_at(arr, 5) is None  # 越界


def test_compute_factor_series_unknown():
    arr = {"close": np.array([1.0, 2.0])}
    assert nf.compute_factor_series("nonexistent", arr) is None


# ---------------- 一致性测试：逐因子 ----------------

@pytest.fixture
def kline_long(rng):
    """长 K 线（150 根），覆盖 long 周期因子 dist_high(n=240 会跳过，用 n=60）。"""
    rets = rng.standard_normal(150) * 0.02
    closes = 10.0 * np.cumprod(1 + rets)
    return [
        {"date": f"2024-{i // 30 + 1:02d}-{i % 30 + 1:02d}",
         "open": float(c) - 0.01, "close": float(c), "high": float(c) + 0.02,
         "low": float(c) - 0.03, "volume": 1000 + int(abs(rets[i]) * 10000)}
        for i, c in enumerate(closes)
    ]


def _assert_consistent(factor_key: str, kline, n, *, atol=1e-9, smooth=None):
    """逐时间点对比纯 Python factors.factor_X 与 numpy compute_factor_series。

    统一通过 compute_factor_series 取 numpy 序列，消除分签名分歧。
    前若干项双方都应为 None/NaN（历史不足），跳过；仅比较两边都有效的点。
    """
    import app.factors as F
    py_fn = getattr(F, f"factor_{factor_key}", None)
    assert callable(py_fn), f"无纯 Python 因子 factor_{factor_key}"

    arr = nf.kline_to_arrays(kline)
    np_series = nf.compute_factor_series(factor_key, arr)
    mismatches = []
    for i in range(len(kline)):
        try:
            if smooth is not None:
                py_val = py_fn(kline, i, n=n, smooth=smooth)
            else:
                py_val = py_fn(kline, i, n=n)
        except TypeError:
            py_val = py_fn(kline, i)
        np_val = nf.series_at(np_series, i)
        if py_val is None:
            continue  # py 历史不足，跳过
        if np_val is None or (isinstance(np_val, float) and np.isnan(np_val)):
            continue  # numpy 历史不足，跳过
        if abs(py_val - np_val) > atol:
            mismatches.append((i, py_val, np_val, f"diff={abs(py_val-np_val):.2e}"))
    assert not mismatches, f"{factor_key} 纯 Python vs numpy 不一致: {mismatches[:5]}"


def test_momentum_consistency(kline_long):
    _assert_consistent("momentum", kline_long, n=20)


def test_ma_dev_consistency(kline_long):
    _assert_consistent("ma_dev", kline_long, n=20)


def test_volatility_consistency(kline_long):
    _assert_consistent("volatility", kline_long, n=20)


def test_rsi_consistency(kline_long):
    _assert_consistent("rsi", kline_long, n=14)


def test_boll_pct_consistency(kline_long):
    _assert_consistent("boll_pct", kline_long, n=20)


def test_amplitude_consistency(kline_long):
    _assert_consistent("amplitude", kline_long, n=20)


def test_wr_consistency(kline_long):
    _assert_consistent("wr", kline_long, n=14)


def test_high_low_pos_consistency(kline_long):
    _assert_consistent("high_low_pos", kline_long, n=20)


def test_volume_ratio_consistency(kline_long):
    _assert_consistent("volume_ratio", kline_long, n=5)


def test_obv_trend_consistency(kline_long):
    _assert_consistent("obv_trend", kline_long, n=20)


def test_dist_high_consistency(kline_long):
    _assert_consistent("dist_high", kline_long, n=60)


def test_cci_consistency(kline_long):
    _assert_consistent("cci", kline_long, n=14)


def test_kdj_k_consistency(kline_long):
    _assert_consistent("kdj_k", kline_long, n=9, smooth=3)


def test_macd_consistency(kline_long):
    """MACD 一致性（周期较长，tol 略宽）。"""
    py_val = factors.factor_macd(kline_long, len(kline_long) - 1)
    arr = nf.kline_to_arrays(kline_long)
    np_series = nf.macd_series(arr["close"])
    np_val = nf.series_at(np_series, len(kline_long) - 1)
    if py_val is None or np_val is None:
        assert py_val is None and (np_val is None or np.isnan(np_val))
    else:
        assert abs(py_val - np_val) < 1e-6, f"MACD diff={abs(py_val-np_val):.2e}"


# ---------------- 扩充因子一致性（逐点版 vs 序列版） ----------------

def test_momentum30_consistency(kline_long):
    """30 日动量：逐点 factor_momentum(30) vs numpy momentum_series(30)。"""
    import app.factors as F
    close = nf.kline_to_arrays(kline_long)["close"]
    np_series = nf.momentum_series(close, 30)
    for i in range(30, len(kline_long)):
        py = F.factor_momentum(kline_long, i, 30)
        npv = nf.series_at(np_series, i)
        if py is None or npv is None:
            continue
        assert abs(py - npv) < 1e-9, (i, py, npv)


def test_momentum90_consistency(kline_long):
    """90 日动量：逐点 factor_momentum(90) vs numpy momentum_series(90)。"""
    import app.factors as F
    close = nf.kline_to_arrays(kline_long)["close"]
    np_series = nf.momentum_series(close, 90)
    for i in range(90, len(kline_long)):
        py = F.factor_momentum(kline_long, i, 90)
        npv = nf.series_at(np_series, i)
        if py is None or npv is None:
            continue
        assert abs(py - npv) < 1e-9, (i, py, npv)


def test_mom_accel_consistency(kline_long):
    _assert_consistent("mom_accel", kline_long, n=20)


def test_ma_align_consistency(kline_long):
    _assert_consistent("ma_align", kline_long, n=60)


def test_ema_slope_consistency(kline_long):
    _assert_consistent("ema_slope", kline_long, n=20)


def test_bias_consistency(kline_long):
    _assert_consistent("bias", kline_long, n=60)


def test_donchian_break_consistency(kline_long):
    _assert_consistent("donchian_break", kline_long, n=20)


def test_dist_low_consistency(kline_long):
    _assert_consistent("dist_low", kline_long, n=60)


def test_skew_consistency(kline_long):
    _assert_consistent("skew", kline_long, n=20)


def test_kurt_consistency(kline_long):
    _assert_consistent("kurt", kline_long, n=20)


def test_down_vol_consistency(kline_long):
    _assert_consistent("down_vol", kline_long, n=20)


def test_max_drawdown_consistency(kline_long):
    _assert_consistent("max_drawdown", kline_long, n=60)


def test_vol_corr_consistency(kline_long):
    _assert_consistent("vol_corr", kline_long, n=20)


def test_vol_change_consistency(kline_long):
    _assert_consistent("vol_change", kline_long, n=5)


def test_catalog_factor_has_numpy_impl():
    """FACTORS 全部技术因子都必须有向量化序列实现（缺失会导致 ML 数据集全空）。"""
    import app.factors as F
    rng = np.random.default_rng(1)
    arr = {"close": rng.random(10) + 1, "high": rng.random(10) + 1,
           "low": rng.random(10) + 1, "volume": rng.random(10) + 1}
    missing = [k for k in F.FACTORS if nf.compute_factor_series(k, arr) is None]
    assert not missing, f"缺少 numpy 序列实现的因子: {missing}"
