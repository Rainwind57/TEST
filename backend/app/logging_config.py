"""结构化日志配置（P2-11）。

旧版各模块用 logging.getLogger(__name__) 但无统一配置，日志格式散乱、
无 request_id 串联、生产环境难以接入 ELK/Loki。本模块提供 JSON 结构化日志，
每条日志含 timestamp/level/logger/message + context（user_id/request_id 等）。
"""
import json
import logging
import sys
import datetime


class JsonFormatter(logging.Formatter):
    """JSON 单行格式，便于日志聚合系统解析。"""

    def format(self, record: logging.LogRecord) -> str:
        # 内置字段
        log = {
            "ts": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # 异常信息
        if record.exc_info:
            log["exc"] = self.formatException(record.exc_info)
        # 业务上下文字段（从 record.__dict__ 取，兼容任意 extra key）
        ctx_keys = ("user_id", "request_id", "job_id", "code", "task")
        for key in ctx_keys:
            val = record.__dict__.get(key)
            if val is not None:
                log[key] = val
        # 兜底：record.__dict__ 里其余非标准 key 也带上（如调用方传了自定义 ctx）
        std_keys = {"name", "msg", "levelname", "levelno", "pathname", "filename",
                    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                    "created", "msecs", "relativeCreated", "thread", "threadName",
                    "processName", "process", "args", "message", "asctime",
                    "taskName", "ts", "level", "logger", "file", "exc"}
        for k, v in record.__dict__.items():
            if k not in std_keys and k not in ctx_keys and not k.startswith("_") and v is not None:
                log[k] = v
        # 模块行号（debug 用）
        if record.levelno <= logging.DEBUG:
            log["file"] = f"{record.filename}:{record.lineno}"
        return json.dumps(log, ensure_ascii=False)


def setup_logging(level: str = "INFO"):
    """配置根 logger：JSON 格式 + stdout 输出（容器友好）。"""
    root = logging.getLogger()
    # 清除既有 handler（避免重复输出）
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # 降低噪音库的日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


class ContextLoggerAdapter(logging.LoggerAdapter):
    """支持调用时传 extra 的 adapter（标准 LoggerAdapter 会丢弃非标准 extra）。

    adapter 自身的 extra（request_id/user_id）与每次调用传的 extra 合并后，
    通过 Logger._log 的 extra 参数注入 record，formatter 再 getattr 取出。
    """

    def process(self, msg, kwargs):
        # 合并 adapter extra 与本次调用 extra（调用 extra 优先），保留 exc_info 等标准参数
        merged = {**(self.extra or {}), **(kwargs.pop("extra", {}) or {})}
        kwargs["extra"] = merged
        return msg, kwargs


def get_request_logger(request_id: str | None = None, user_id: int | None = None):
    """获取带上下文的 logger adapter（HTTP 请求/任务用）。"""
    logger = logging.getLogger("quant.request")
    extra = {}
    if request_id:
        extra["request_id"] = request_id
    if user_id:
        extra["user_id"] = user_id
    return ContextLoggerAdapter(logger, extra)
