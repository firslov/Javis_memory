"""FastAPI 应用入口

配置应用启动、关闭事件和路由注册。
"""
import asyncio
from pathlib import Path
from typing import Dict
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from api.routes.chat import router as chat_router
from api.routes.api_keys import router as api_keys_router
from api.routes.memory import router as memory_router
from api.middleware import RequestLoggingMiddleware
from database.session import init_db, close_db
from services.llm_forwarder import LLMForwarder
from services.memory_index import MemoryIndexManager
from services.file_watcher import start_memory_file_watcher, stop_all_file_watchers
from config.settings import get_settings
from config.logging import setup_logging, get_logger


# 创建FastAPI应用
app = FastAPI(
    title="Javis",
    description="个人AI助手系统 - RAG向量检索记忆",
    version="0.2.0",
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加日志中间件
app.add_middleware(RequestLoggingMiddleware)


# ============================================================================
# 全局异常处理器
# ============================================================================

@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    """处理数据库相关异常"""
    from config.logging import get_logger
    logger = get_logger(__name__)

    logger.error(f"[DB] Database error on {request.url.path}: {exc}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Database error",
            "message": "数据库操作失败。请确保已运行 `python init_db.py` 初始化数据库。",
            "detail": str(exc) if logger.level <= 10 else None  # DEBUG 模式才显示详情
        }
    )


# 注册路由
app.include_router(chat_router)
app.include_router(api_keys_router)
app.include_router(memory_router)


# Global file watcher reference
_file_watcher = None

# Global task scheduler reference
_task_scheduler = None

# Logger
logger = get_logger(__name__)


@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    global _file_watcher

    try:
        settings = get_settings()
    except Exception as e:
        print("配置加载失败！")
        print(f"错误: {e}")
        print()
        print("请检查 config/servers.yaml 文件:")
        print("  1. 文件是否存在？")
        print("  2. YAML 格式是否正确？")
        print("  3. 是否填写了 API Key？")
        raise

    # Setup centralized logging
    log_file = Path("logs/javis.log") if settings.logging.level == "DEBUG" else None
    setup_logging(
        level=settings.logging.level,
        log_file=log_file
    )

    # Set main event loop for file watcher
    from services.file_watcher import set_main_event_loop
    set_main_event_loop(asyncio.get_event_loop())

    logger.info("[SERVER] Starting Javis...")
    logger.info(f"[SERVER] Memory system: {'enabled' if settings.memory_search.enabled else 'disabled'}")

    # 1. Initialize database
    try:
        await init_db()
        logger.info("[SERVER] Database initialized")
    except Exception as e:
        logger.error(f"[SERVER] Database initialization failed: {e}")
        print()
        print("数据库初始化失败！")
        print("请先运行: python init_db.py")
        raise

    # 2. Initialize LLM forwarder and store to app.state
    try:
        llm_forwarder = LLMForwarder()
        await llm_forwarder.initialize()
        app.state.llm_forwarder = llm_forwarder
        model_count = len(llm_forwarder.get_available_models())
        logger.info(f"[SERVER] LLM forwarder ready ({model_count} models)")
    except Exception as e:
        logger.error(f"[SERVER] LLM forwarder initialization failed: {e}")
        print()
        print("LLM 转发器初始化失败！")
        print("请检查 config/servers.yaml 中的服务器配置是否正确")
        raise

    # 3. Initialize memory managers container
    app.state.memory_managers: Dict[int, MemoryIndexManager] = {}

    # 4. Initialize file watcher for memory sync
    if settings.memory_search.enabled and settings.memory_search.sync.watch:
        try:
            _file_watcher = await start_memory_file_watcher(
                config=settings.memory_search,
                sync_callback=_sync_all_memories,
            )
            if _file_watcher:
                logger.info("[WATCH] File watcher started for memory sync")
        except Exception as e:
            logger.warning(f"[WATCH] File watcher failed to start: {e}")
            logger.warning("[WATCH] Memory auto-sync disabled, but manual sync still works")

    # 5. Initialize task scheduler for memory lifecycle management
    global _task_scheduler
    if settings.memory_search.enabled:
        lifecycle_enabled = settings.memory_search.lifecycle.enabled
        consolidation_enabled = settings.memory_search.consolidation.enabled

        if lifecycle_enabled or consolidation_enabled:
            try:
                from services.scheduler import MemoryTaskScheduler, set_scheduler

                _task_scheduler = MemoryTaskScheduler(settings.memory_search)

                # 定义获取管理器的函数
                async def get_manager(user_id: int):
                    return await _get_or_create_manager(user_id)

                await _task_scheduler.start(get_manager)
                set_scheduler(_task_scheduler)
                logger.info("[SCHEDULER] Task scheduler started for memory lifecycle")
            except Exception as e:
                logger.warning(f"[SCHEDULER] Failed to start task scheduler: {e}")
                logger.warning("[SCHEDULER] Automated lifecycle management disabled")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的清理"""
    global _file_watcher, _task_scheduler

    logger.info("[SERVER] Shutting down...")

    # Stop task scheduler
    if _task_scheduler:
        try:
            await _task_scheduler.stop()
            logger.info("[SCHEDULER] Task scheduler stopped")
        except Exception as e:
            logger.error(f"[SCHEDULER] Error stopping scheduler: {e}")

    # Stop file watchers
    await stop_all_file_watchers()

    # Close all memory managers
    for manager in getattr(app.state, "memory_managers", {}).values():
        await manager.close()
    app.state.memory_managers = {}

    # Close LLM forwarder
    if hasattr(app.state, "llm_forwarder"):
        await app.state.llm_forwarder.close()

    # Close database
    await close_db()
    logger.info("[SERVER] Shutdown complete")


async def _sync_all_memories():
    """Sync all user memories (callback for file watcher)"""
    managers: Dict[int, MemoryIndexManager] = app.state.memory_managers

    for user_id, manager in list(managers.items()):
        try:
            await manager.sync()
            logger.debug(f"[MEMORY] Auto-synced for user {user_id}")
        except Exception as e:
            logger.error(f"[MEMORY] Auto-sync failed for user {user_id}: {e}")


async def _get_or_create_manager(user_id: int) -> MemoryIndexManager:
    """Get or create a memory index manager for a user.

    This helper function is used by the task scheduler to access managers.
    """
    from api.routes.dependencies import get_or_create_memory_manager

    # 创建一个简单的请求对象用于依赖注入
    class DummyRequest:
        pass

    dummy_request = DummyRequest()
    dummy_request.app = app

    return await get_or_create_memory_manager(dummy_request, user_id)


@app.get("/")
async def root():
    """服务信息"""
    return {
        "name": "Javis",
        "version": "0.2.0",
        "memory_system": "RAG Vector Search",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy", "memory_system": "RAG"}
