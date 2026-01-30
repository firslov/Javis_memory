"""共享的依赖注入函数

提供 FastAPI 路由使用的依赖注入函数，用于注入 LLMForwarder 和 MemoryIndexManager 等服务。
"""
from typing import Dict

from fastapi import Request

from services.llm_forwarder import LLMForwarder
from services.memory_index import MemoryIndexManager
from config.settings import get_settings


async def get_llm_forwarder(request: Request) -> LLMForwarder:
    """Get LLM forwarder from app state (dependency injection).

    Args:
        request: FastAPI Request object.

    Returns:
        LLMForwarder instance.
    """
    return request.app.state.llm_forwarder


async def get_or_create_memory_manager(
    request: Request,
    user_id: int,
) -> MemoryIndexManager:
    """Get or create a memory index manager for a user (dependency injection).

    This function creates managers on-demand and stores them in app.state
    for reuse across requests.

    Args:
        request: FastAPI Request object.
        user_id: User ID for the memory manager.

    Returns:
        MemoryIndexManager instance for the user.
    """
    settings = get_settings()
    managers: Dict[int, MemoryIndexManager] = request.app.state.memory_managers

    if user_id not in managers:
        db_path = settings.memory_search.store.path.format(user_id=user_id)
        manager = MemoryIndexManager(user_id, db_path, settings.memory_search)
        await manager.initialize()
        managers[user_id] = manager

    return managers[user_id]
