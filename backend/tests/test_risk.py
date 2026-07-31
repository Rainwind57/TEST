"""risk.py 风险模块单元测试（P2-13）。

覆盖：
- factor_covariance 输入校验（空/单行）
- cross_section_regression OLS 正确性
- attribute_returns 收益归因恒等式
- risk_decomposition 风险分解非负性
- estimate_specific_variances 逐股特质方差（P1-8）
- value_at_risk / conditional_var（P1-8）
"""
import numpy as np
import pytest

from app import risk


def test_factor_covariance_empty():
    assert risk.factor_covariance(np.zeros((0, 0))).size == 0


def test_factor_covariance_single_row():
    """单行输入返回零矩阵（样本不足）。"""
    cov = risk.factor_covariance(np.array([[1.0, 2.0, 3.0]]))
    assert cov.shape == (3, 3)
    assert np.all(cov == 0)


def test_factor_covariance_symmetric():
    """协方差矩阵对称。"""
    np.random.seed(0)
    panel = np.random.randn(20, 3)
    cov = risk.factor_covariance(panel)
    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T)


def test_cross_section_regression_ols():
    """OLS 截面回归：y = 2*x1 + 0.5*x2，系数方向正确。"""
    np.random.seed(0)
    X = np.random.randn(50, 2)
    beta = np.array([2.0, 0.5])
    y = X @ beta + 1.0  # 加截距
    est = risk.cross_section_regression(X, y)
    assert est[0] == pytest.approx(2.0, rel=0.1)
    assert est[1] == pytest.approx(0.5, rel=0.1)


def test_cross_section_regression_empty():
    assert risk.cross_section_regression(np.zeros((0, 0)), np.array([])).size == 0


def test_attribute_returns_identity():
    """收益归因恒等式：组合收益 = 因子贡献 + 残差。"""
    np.random.seed(0)
    n_stocks, n_factors = 10, 3
    X = np.random.randn(n_stocks, n_factors)
    w = np.ones(n_stocks) / n_stocks
    fr = np.random.randn(n_factors)
    sr = X @ fr + np.random.randn(n_stocks) * 0.1
    r = risk.attribute_returns(w, X, fr, sr)
    assert abs(r["total"] - r["totalFactor"] - r["residual"]) < 1e-9


def test_risk_decomposition_nonnegative():
    """风险分解各项非负（方差开根）。"""
    np.random.seed(0)
    X = np.random.randn(10, 3)
    w = np.ones(10) / 10
    sf = np.cov(np.random.randn(20, 3), rowvar=False)
    se = np.abs(np.random.randn(10))
    r = risk.risk_decomposition(w, X, sf, se)
    assert r["factorRisk"] >= 0
    assert r["specificRisk"] >= 0
    assert r["totalRisk"] >= 0
    # totalRisk² ≈ factorRisk² + specificRisk²
    assert r["totalRisk"] ** 2 == pytest.approx(
        r["factorRisk"] ** 2 + r["specificRisk"] ** 2, rel=1e-6)


def test_risk_decomposition_empty():
    r = risk.risk_decomposition(np.array([]), np.zeros((0, 0)), np.zeros((0, 0)), np.array([]))
    assert r == {"factorRisk": 0.0, "specificRisk": 0.0, "totalRisk": 0.0}


def test_value_at_risk_basic():
    """历史模拟 VaR：已知分布的分位数。"""
    np.random.seed(0)
    rets = np.random.randn(1000) * 0.01  # 标准正态*0.01
    r = risk.value_at_risk(rets, alpha=0.05)
    # 95% VaR ≈ 1.645 * 0.01 ≈ 0.01645
    assert r["var"] == pytest.approx(0.0164, abs=0.005)
    assert r["cvar"] > r["var"]  # CVaR 必大于 VaR（尾部均值比分位深）
    assert r["n"] == 1000


def test_value_at_risk_insufficient_samples():
    """样本不足返回告警而非崩。"""
    r = risk.value_at_risk(np.array([0.01, 0.02]), alpha=0.05)
    assert r["var"] == 0.0
    assert "warning" in r


def test_value_at_risk_with_weights():
    """多资产矩阵 + 权重 → 组合收益 VaR。"""
    np.random.seed(0)
    rets = np.random.randn(100, 5)
    w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    r = risk.value_at_risk(rets, weights=w, alpha=0.05)
    assert r["n"] == 100
    assert r["var"] > 0


def test_conditional_var_numeric():
    np.random.seed(0)
    rets = -np.abs(np.random.randn(100))  # 全负收益
    cvar = risk.conditional_var(rets, alpha=0.1)
    assert cvar > 0


def test_estimate_specific_variances_per_stock():
    """逐股方差应不同（非旧版标量广播），样本不足股用截面均值兜底。"""
    np.random.seed(0)
    n = 120
    mk = lambda: (np.cumprod(1 + np.random.randn(n) * 0.02 + 0.001) * 10).tolist()
    sd = [
        {"code": f"c{i}", "quote": {"turnover": 1.0, "pe": 15.0 + i, "pb": 2.0},
         "kline": [{"date": f"2024-{j // 28 + 1:02d}-{j % 28 + 1:02d}",
                    "open": c, "close": c, "high": c, "low": c, "volume": 1000}
                   for j, c in enumerate(mk())]}
        for i in range(3)
    ]
    X = np.random.randn(3, 5)
    codes = ["c0", "c1", "c2"]
    fr = np.random.randn(5)
    se = risk.estimate_specific_variances(sd, X, codes, fr)
    assert len(se) == 3
    assert np.all(se >= 0)


def test_estimate_specific_variances_empty():
    """空输入返回空数组（不崩）。"""
    se = risk.estimate_specific_variances([], np.zeros((0, 0)), [], np.array([]))
    assert se.size == 0
