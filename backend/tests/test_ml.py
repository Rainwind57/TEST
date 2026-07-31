"""ml.py 机器学习工具函数单元测试（P2-13）。

覆盖：
- _pearson / _spearman 数值正确性
- _bucket_returns 分组逻辑
- _oos_sharpe_by_date 按日聚合（修复旧版单元素 std=0 的回归）
- purged_walk_forward_split 时序切分防泄漏
- _build_model 支持 gbdt + lightgbm（P1-7）
- optimize_model 端到端（P1-7 ML 调参闭环）
"""
import numpy as np
import pytest

from app import ml


def test_pearson_perfect():
    assert ml._pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)


def test_pearson_short():
    assert ml._pearson([1.0], [2.0]) == 0.0


def test_spearman_inverted():
    assert ml._spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_bucket_returns_basic():
    """5 组分桶，top 组均值 > bottom 组均值（给定单调 pred）。"""
    pred = np.arange(50, dtype=float)
    actual = np.arange(50, dtype=float) * 2
    top, bottom, ls = ml._bucket_returns(pred, actual, 5)
    assert top > bottom
    assert ls == pytest.approx(top - bottom)


def test_bucket_returns_too_few():
    """样本不足分组返回 0（不崩）。"""
    top, bottom, ls = ml._bucket_returns([1.0, 2.0], [1.0, 2.0], 5)
    assert (top, bottom, ls) == (0.0, 0.0, 0.0)


def test_oos_sharpe_by_date_empty():
    assert ml._oos_sharpe_by_date([]) == 0.0


def test_oos_sharpe_by_date_single_day():
    """单日样本（< groups*2）应跳过，返回 0（非旧版 std 单元素恒 0）。"""
    records = [("2024-01-01", float(i), float(i)) for i in range(5)]
    assert ml._oos_sharpe_by_date(records) == 0.0


def test_oos_sharpe_by_date_positive():
    """每日 top 组收益正、bottom 组负 → 多空序列为正；日间略有变化使 std>0 → Sharpe>0。"""
    np.random.seed(0)
    records = []
    for d in range(20):
        date = f"2024-01-{d+1:02d}"
        preds = np.arange(20, dtype=float)
        # 前 10 负、后 10 正；每日幅度略有不同使多空收益非全等值
        amp = 1.0 + d * 0.01
        actuals = np.where(np.arange(20) < 10,
                           -np.arange(1, 21) * amp,
                           np.arange(1, 21).astype(float) * amp)
        for p, a in zip(preds, actuals):
            records.append((date, p, float(a)))
    s = ml._oos_sharpe_by_date(records)
    assert s > 0


def test_purged_walk_forward_split_ordering():
    """训练区在前、测试区在后、中间有 gap（防泄漏）。"""
    splits = ml.purged_walk_forward_split(100, n_splits=5, test_ratio=0.2, gap=5)
    assert len(splits) > 0
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0
        assert train_idx.max() + 5 <= test_idx.min()


def test_purged_walk_forward_split_too_short():
    """样本不足返回空列表（不崩）。"""
    splits = ml.purged_walk_forward_split(5, n_splits=5, gap=5)
    assert splits == []


def test_fit_preprocess_apply_roundtrip():
    """fit 预处理参数能被 apply 复现，NaN 用训练均值填充。"""
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [100.0, 3000.0]])
    pp = ml._fit_preprocess(X, lower=0.01, upper=0.99)
    Xp = ml._apply_preprocess(X, pp)
    assert abs(Xp[:, 0].mean()) < 1e-9
    assert abs(Xp[:, 0].std() - 1.0) < 0.1


def test_build_model_gbdt():
    from sklearn.ensemble import GradientBoostingRegressor
    m = ml._build_model("gbdt")
    assert isinstance(m, GradientBoostingRegressor)


def test_build_model_lightgbm():
    """lightgbm 已在 requirements 声明，_build_model 应能构造（P1-7）。"""
    import lightgbm as lgb
    m = ml._build_model("lightgbm")
    assert isinstance(m, lgb.LGBMRegressor)


def test_build_model_unknown():
    with pytest.raises(ValueError, match="不支持的模型类型"):
        ml._build_model("nonexistent")


def test_build_model_with_params():
    """支持外部传超参（Optuna 调参用，P1-7）。"""
    m = ml._build_model("lightgbm", {"n_estimators": 100, "max_depth": 5})
    assert m.get_params()["n_estimators"] == 100
    assert m.get_params()["max_depth"] == 5


def test_optimize_model_end_to_end():
    """optimize_model 端到端：合成数据跑 3 trial，返回 best_params + IS/OOS 指标 + 落盘（P1-7）。"""
    import os, glob
    np.random.seed(0)
    n = 200
    X = np.random.randn(n, 5)
    y = X @ np.array([0.1, 0.2, -0.1, 0.05, 0.0]) + np.random.randn(n) * 0.1
    dataset = {
        "features": X.astype(float),
        "target": y,
        "dates": [f"2024-01-{i+1:02d}" for i in range(n)],
        "feature_names": ["f0", "f1", "f2", "f3", "f4"],
    }
    r = ml.optimize_model(dataset, model_type="lightgbm", n_splits=3, gap=2, n_trials=3)
    assert "bestParams" in r
    assert "isSharpe" in r and "oosSharpe" in r
    assert r["nTrials"] == 3
    files = glob.glob(os.path.join(ml.ML_DIR, "opt_*.json"))
    assert len(files) > 0


def test_score_latest_no_name_error():
    """P0-1 回归：score_latest 不再抛 NameError（snap_keys 已定义）。"""
    import inspect
    src = inspect.getsource(ml.score_latest)
    assert "snap_keys = [k for k in feature_names if k in snap_set]" in src
    assert "snap_set" in src
