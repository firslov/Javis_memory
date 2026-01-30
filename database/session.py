"""数据库会话管理

提供数据库引擎、会话工厂和异步会话生成器。
"""
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import text
from config.settings import get_settings

from .models import Base


# 全局引擎和会话工厂
_engine = None
_async_session_factory = None


def get_engine():
    """获取数据库引擎"""
    global _engine
    if _engine is None:
        settings = get_settings()

        # SQLite 特殊配置，提高并发性能
        engine_kwargs = {}
        if settings.database.url.startswith("sqlite"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}

        _engine = create_async_engine(
            settings.database.url,
            echo=False,
            **engine_kwargs
        )

    return _engine


def get_session_factory():
    """获取会话工厂"""
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """获取数据库会话（依赖注入用）"""
    async_session = get_session_factory()
    async with async_session() as session:
        # 为每个连接启用外键约束和设置超时 (SQLite需要)
        if session.bind.dialect.name == "sqlite":
            await session.execute(text("PRAGMA foreign_keys=ON"))
            # 设置30秒的busy_timeout，等待数据库解锁
            await session.execute(text("PRAGMA busy_timeout=30000"))
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """初始化数据库表"""
    engine = get_engine()

    # 启用 WAL 模式以提高并发性能
    if engine.dialect.name == "sqlite":
        async with engine.connect() as conn:
            await conn.execute(text("PRAGMA journal_mode=WAL"))
            await conn.execute(text("PRAGMA foreign_keys=ON"))
            await conn.commit()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """关闭数据库连接"""
    global _engine, _async_session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _async_session_factory = None
