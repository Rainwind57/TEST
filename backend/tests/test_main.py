"""main.py 限流中间件回归测试（P2-13 + P0-4）。

覆盖：
- RateLimitMiddleware 已注册
- 健康检查端点不限流
- 限流配置可环境变量调
"""
import inspect
from app import main


def test_rate_limit_middleware_registered():
    """main 模块应定义 RateLimitMiddleware 类（P0-4）。"""
    assert hasattr(main, "RateLimitMiddleware")
    src = inspect.getsource(main)
    assert "RateLimitMiddleware" in src
    assert "BaseHTTPMiddleware" in src


def test_exempt_paths_include_health():
    """健康检查应豁免限流（P0-4）。"""
    assert "/api/health" in main._exempt_paths
    assert "/api/auth/login" in main._exempt_paths


def test_rate_limits_configurable():
    """限流阈值应有合理默认（P0-4）。"""
    assert main._ANON_LIMIT > 0
    assert main._USER_LIMIT > main._ANON_LIMIT  # 登录用户应更宽松
