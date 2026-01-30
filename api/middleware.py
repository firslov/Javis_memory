"""请求监控和日志中间件

记录 API 请求的处理时间和错误信息。
"""
import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import get_settings
from config.logging import get_logger


logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """简化的请求日志中间件"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理请求并记录日志"""
        settings = get_settings()
        start_time = time.time()

        # 处理请求
        try:
            response = await call_next(request)
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"[API] {request.method} {request.url.path} - {process_time:.3f}s - ERROR: {str(e)}",
                extra={"path": str(request.url.path), "method": request.method, "status": "error"}
            )
            raise

        # 计算处理时间
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.3f}"

        # 只记录非200的响应或慢请求
        if response.status_code >= 400:
            logger.warning(
                f"[API] {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s",
                extra={"path": str(request.url.path), "method": request.method, "status": response.status_code}
            )
        elif settings.logging.slow_request_threshold and process_time > settings.logging.slow_request_threshold:
            logger.warning(
                f"[API] {request.method} {request.url.path} - SLOW - {process_time:.3f}s",
                extra={"path": str(request.url.path), "method": request.method, "slow": True}
            )

        return response
