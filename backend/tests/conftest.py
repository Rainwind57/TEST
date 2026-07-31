"""pytest 共享配置：把 backend/ 加入 sys.path 使 tests 可 import app 包。"""
import sys
import os

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import pytest
import numpy as np


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def kline_uptrend():
    """60 根单边上扬 K 线（确定性序列，便于断言 momentum>0、rsi>50）。"""
    import numpy as np
    base = 10.0
    closes = [base * (1.001 ** i) + np.sin(i / 5) * 0.05 for i in range(60)]
    return [
        {"date": f"2024-01-{i // 28 + 1:02d}-{i % 28 + 1:02d}",
         "open": c - 0.01, "close": c, "high": c + 0.02, "low": c - 0.03,
         "volume": 1000 + i * 10}
        for i, c in enumerate(closes)
    ]


@pytest.fixture
def kline_random(rng):
    """120 根随机游走 K 线（足够长度覆盖长周期因子）。"""
    import numpy as np
    rets = rng.standard_normal(120) * 0.02
    closes = 10.0 * np.cumprod(1 + rets)
    return [
        {"date": f"2024-{i // 30 + 1:02d}-{i % 30 + 1:02d}",
         "open": c - 0.01, "close": float(c), "high": float(c) + 0.02,
         "low": float(c) - 0.03, "volume": 1000 + int(abs(rets[i]) * 10000)}
        for i, c in enumerate(closes)
    ]
