"""logging_config.py 结构化日志单元测试（P2-11 + P2-13）。"""
import json
import logging
import io
import pytest

from app import logging_config


@pytest.fixture
def capture_logs():
    """捕获日志输出到 StringIO。"""
    buf = io.StringIO()
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    h = logging.StreamHandler(buf)
    h.setFormatter(logging_config.JsonFormatter())
    root.handlers = [h]
    yield buf
    root.handlers = old_handlers


def test_setup_logging_sets_level():
    logging_config.setup_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG
    logging_config.setup_logging("INFO")


def test_json_format_basic(capture_logs):
    log = logging_config.get_request_logger()
    log.info("hello")
    line = capture_logs.getvalue().strip()
    data = json.loads(line)
    assert data["level"] == "INFO"
    assert data["msg"] == "hello"
    assert "ts" in data
    assert "logger" in data


def test_json_format_with_context(capture_logs):
    log = logging_config.get_request_logger(request_id="req-1", user_id=42)
    log.info("with ctx")
    data = json.loads(capture_logs.getvalue().strip())
    assert data["request_id"] == "req-1"
    assert data["user_id"] == 42


def test_json_format_exception(capture_logs):
    log = logging_config.get_request_logger()
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("err")
    data = json.loads(capture_logs.getvalue().strip())
    assert "exc" in data
    assert "ValueError" in data["exc"]


def test_logger_adapter_merges_extra(capture_logs):
    """LoggerAdapter 合并 adapter extra 与调用 extra。"""
    log = logging_config.get_request_logger(request_id="r1", user_id=1)
    log.info("msg", extra={"job_id": "j1"})
    data = json.loads(capture_logs.getvalue().strip())
    assert data["request_id"] == "r1"
    assert data["user_id"] == 1
    assert data["job_id"] == "j1"


def test_noise_libs_silenced():
    logging_config.setup_logging("INFO")
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("apscheduler").level == logging.WARNING
