"""量化平台测试套件（P2-13）。

覆盖：
- factors.py 纯 Python 因子 + 统计/回归函数单元测试
- numpy_factors.py 向量化因子 + 与 factors.py 同因子数值一致性测试
- ml.py 机器学习工具函数（pearson/spearman/bucket_returns/oos_sharpe/walk_forward split）
- risk.py 风险分解 + VaR/CVaR + 特质方差估计
- scheduler/ml 修复回归（P0 snap_keys、scheduler 交易日历）
"""
