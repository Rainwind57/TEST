import sys
sys.path.insert(0, '.')
import numpy as np
from app.factors import FACTORS

# 模拟腾讯 K 线结构：dict 列表，含 date/open/close/high/low/volume
rets = np.random.default_rng(7).standard_normal(260) * 0.02
closes = 10.0 * np.cumprod(1 + rets)
kline = [
    {"date": f"2024-01-{i % 28 + 1:02d}", "open": float(c) - 0.01, "close": float(c),
     "high": float(c) + 0.02, "low": float(c) - 0.03, "volume": 1000 + int(abs(rets[i]) * 10000)}
    for i, c in enumerate(closes)
]
i = len(kline) - 1
errs = []
vals = {}
for key, meta in FACTORS.items():
    try:
        vals[key] = meta["calc"](kline, i)
    except Exception as e:
        errs.append((key, repr(e)))
print("total factors:", len(FACTORS))
print("errors:", errs if errs else "NONE")
print("all valid:", {k: round(v, 4) for k, v in vals.items() if v is not None})
print("None count:", sum(1 for v in vals.values() if v is None), "/", len(vals))
