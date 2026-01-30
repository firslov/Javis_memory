"""集中式日志配置

为所有服务提供统一的日志格式和行为。

特性:
    - 彩色日志输出
    - 统一的日志格式
    - 自动日志轮转（10MB，保留5个备份）
    - 过滤冗余日志（httpx、uvicorn）
"""
import logging
import logging.handlers
import sys
from typing import Optional
from pathlib import Path


# ============================================================================
# Color Support
# ============================================================================

class LogColors:
    """日志颜色"""
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # Level colors
    DEBUG = '\033[36m'      # Cyan
    INFO = '\033[37m'       # White
    WARNING = '\033[33m'    # Yellow
    ERROR = '\033[31m'      # Red
    CRITICAL = '\033[35m'   # Magenta

    # Module colors
    SERVER = '\033[96m'     # Bright Cyan
    API = '\033[94m'        # Blue
    MEMORY = '\033[92m'     # Green
    LLM = '\033[95m'        # Magenta
    SEARCH = '\033[93m'     # Yellow

    DISABLED = False


def should_colorize() -> bool:
    """判断是否应该输出颜色"""
    return (
        not LogColors.DISABLED and
        sys.stdout.isatty() and
        hasattr(sys.stdout, 'isatty') and
        sys.stdout.isatty()
    )


# ============================================================================
# Log Format Constants
# ============================================================================

# Color format for console
COLOR_FORMAT = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'

# Plain format for files
PLAIN_FORMAT = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'

# ISO 8601 timestamp format
TIMESTAMP_FORMAT = '%H:%M:%S'


# ============================================================================
# Custom Formatter with Colors
# ============================================================================

class ColorFormatter(logging.Formatter):
    """带颜色的日志格式化器"""

    # 日志级别颜色映射
    LEVEL_COLORS = {
        logging.DEBUG: LogColors.DEBUG,
        logging.INFO: LogColors.INFO,
        logging.WARNING: LogColors.WARNING,
        logging.ERROR: LogColors.ERROR,
        logging.CRITICAL: LogColors.CRITICAL,
    }

    # 模块名称颜色映射
    MODULE_COLORS = {
        'api': LogColors.API,
        'api.main': LogColors.API,
        'api.routes': LogColors.API,
        'services.llm_forwarder': LogColors.LLM,
        'services.memory': LogColors.MEMORY,
        'services.memory_index': LogColors.MEMORY,
        'services.search': LogColors.SEARCH,
        'services.search_engine': LogColors.SEARCH,
    }

    def __init__(self, fmt: str = None, datefmt: str = None):
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        # 获取基础格式化结果
        result = super().format(record)

        if not should_colorize():
            return result

        # 添加颜色
        level_color = self.LEVEL_COLORS.get(record.levelno, LogColors.INFO)
        level_name = f"{level_color}{record.levelname:8}{LogColors.RESET}"

        # 模块名颜色
        module_color = self._get_module_color(record.name)
        module_name = f"{module_color}{record.name:20}{LogColors.RESET}"

        # 时间戳颜色
        timestamp = f"{LogColors.RESET}{record.asctime.split()[1] if len(record.asctime.split()) > 1 else record.asctime}{LogColors.RESET}"

        # 消息内容
        message = record.getMessage()

        # 组合
        return f"{timestamp} | {level_name} | {module_name} | {message}"

    def _get_module_color(self, module_name: str) -> str:
        """获取模块对应的颜色"""
        for key, color in self.MODULE_COLORS.items():
            if module_name.startswith(key):
                return color
        return LogColors.RESET


class PlainFormatter(logging.Formatter):
    """普通日志格式化器（用于文件）"""
    def __init__(self, fmt: str = None, datefmt: str = None):
        super().__init__(fmt, datefmt)


# ============================================================================
# Log Level Configuration
# ============================================================================

def parse_log_level(level_str: str) -> int:
    """Parse log level string to logging constant.

    Args:
        level_str: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Logging level constant.
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_map.get(level_str.upper(), logging.INFO)


# ============================================================================
# Quiet Log Filter
# ============================================================================

class QuietLogFilter(logging.Filter):
    """Filter to suppress specific log messages.

    Filters out:
    - httpx INFO logs (HTTP request logs)
    - uvicorn access logs (handled by middleware)
    """

    def __init__(self):
        super().__init__()
        # Track whether we've logged certain one-time messages
        self._logged_messages = set()

    def filter(self, record: logging.LogRecord) -> bool:
        # Filter out httpx INFO logs (too verbose)
        if record.name == "httpx" and record.levelno <= logging.INFO:
            return False

        # Filter out uvicorn access logs
        if record.name.startswith("uvicorn.access"):
            return False

        # Deduplicate repeated initialization messages
        msg_key = f"{record.name}:{record.msg}"
        if "Services initialized" in record.msg or "FTS5 index created" in record.msg:
            if msg_key in self._logged_messages:
                return False
            self._logged_messages.add(msg_key)

        return True


# ============================================================================
# Logger Configuration
# ============================================================================

def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
) -> None:
    """Configure application-wide logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(parse_log_level(level))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create color formatter for console
    console_formatter = ColorFormatter(
        fmt=COLOR_FORMAT,
        datefmt=TIMESTAMP_FORMAT
    )

    # Create filter
    quiet_filter = QuietLogFilter()

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(parse_log_level(level))
    console_handler.addFilter(quiet_filter)
    root_logger.addHandler(console_handler)

    # Add file handler (optional, for debug mode)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Use plain formatter for files
        file_formatter = PlainFormatter(
            fmt=PLAIN_FORMAT,
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Use rotating file handler: 10MB max, keep 5 backup files
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)

    # Configure third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with consistent configuration.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


# ============================================================================
# Initialization (deferred, will be called by app startup)
# ============================================================================

# Don't auto-initialize on import - let the app control it
# setup_logging(level="INFO")
