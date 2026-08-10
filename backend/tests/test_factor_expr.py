"""factor_expr.py 表达式引擎单元测试（P1-6 + P2-13）。

覆盖：
- 基础算术表达式
- 截面函数 rank/zscore
- 安全校验：拒 __import__/open/属性访问/下标
- 空输入/缺失列容错
- NaN 处理
"""
import math
import pytest
import numpy as np

from app import factor_expr as fe


@pytest.fixture
def rows():
    return [
        {"code": "c1", "momentum": 0.1, "volatility": 0.2, "rsi": 55},
        {"code": "c2", "momentum": 0.3, "volatility": 0.1, "rsi": 45},
        {"code": "c3", "momentum": -0.1, "volatility": 0.15, "rsi": 65},
    ]


def test_basic_arithmetic(rows):
    """momentum - volatility*0.5 逐行算术。"""
    out = fe.evaluate_expression("momentum - volatility * 0.5", rows)
    assert len(out) == 3
    assert out[0] == pytest.approx(0.1 - 0.2 * 0.5)
    assert out[1] == pytest.approx(0.3 - 0.1 * 0.5)
    assert out[2] == pytest.approx(-0.1 - 0.15 * 0.5)


def test_rank_function(rows):
    """rank 归一化到 [0,1]，最大值=1、最小值=0。"""
    out = fe.evaluate_expression("rank(momentum)", rows)
    assert min(out) == 0.0  # 最小 momentum 对应 0
    assert max(out) == 1.0  # 最大 momentum 对应 1


def test_zscore_function(rows):
    """zscore 均值=0。"""
    out = fe.evaluate_expression("zscore(momentum)", rows)
    assert sum(out) == pytest.approx(0.0, abs=1e-9)


def test_combined_expression(rows):
    """复合表达式：rank + zscore。"""
    out = fe.evaluate_expression("rank(momentum) - rank(volatility)", rows)
    assert len(out) == 3
    # c2 momentum 最大、volatility 最小 → rank 差最大
    assert out[1] == max(out)


def test_validate_empty():
    ok, err = fe.validate_expression("")
    assert not ok


def test_validate_syntax_error():
    ok, err = fe.validate_expression("momentum +")
    assert not ok
    assert "语法错误" in err


def test_validate_reject_dunder_import():
    """拒绝 __import__（防任意代码执行）。"""
    ok, err = fe.validate_expression('__import__("os")')
    assert not ok
    assert "__import__" in err


def test_validate_reject_open():
    ok, err = fe.validate_expression('open("x")')
    assert not ok
    assert "open" in err


def test_validate_reject_attribute():
    """拒绝属性访问（防访问对象方法）。"""
    ok, err = fe.validate_expression("a.b")
    assert not ok
    assert "Attribute" in err


def test_validate_reject_subscript():
    """拒绝下标访问。"""
    ok, err = fe.validate_expression("a[0]")
    assert not ok
    assert "Subscript" in err


def test_validate_reject_listcomp():
    ok, err = fe.validate_expression("[x for x in momentum]")
    assert not ok


def test_validate_accept_simple():
    ok, err = fe.validate_expression("momentum + rsi")
    assert ok


def test_validate_accept_safe_funcs():
    ok, _ = fe.validate_expression("max(momentum, 0)")
    assert ok
    ok, _ = fe.validate_expression("abs(momentum)")
    assert ok


def test_empty_rows():
    assert fe.evaluate_expression("momentum", []) == []


def test_missing_column(rows):
    """缺失列视为 0（不崩）。"""
    out = fe.evaluate_expression("nonexistent + momentum", rows)
    assert out == pytest.approx([0.1, 0.3, -0.1])


def test_nan_handled(rows):
    """含 NaN 的列按截面中位数填充（避免单 NaN 把整列传染成 0）。"""
    rows_with_nan = [{"code": "c1", "momentum": float("nan")}] + rows
    out = fe.evaluate_expression("momentum", rows_with_nan)
    # momentum=[NaN,0.1,0.3,-0.1]，中位数=0.1 → c1 填充为 0.1
    assert out[0] == pytest.approx(0.1)
    assert len(out) == 4


def test_constant_expression(rows):
    """常量表达式广播到所有行。"""
    out = fe.evaluate_expression("1.5", rows)
    assert all(v == 1.5 for v in out)


def test_unary_minus(rows):
    out = fe.evaluate_expression("-momentum", rows)
    assert out[0] == pytest.approx(-0.1)


def test_power(rows):
    out = fe.evaluate_expression("momentum ** 2", rows)
    assert out[0] == pytest.approx(0.01)
