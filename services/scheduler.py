"""后台任务调度器

管理记忆系统的定时任务：
- 记忆合并
- 工作记忆清理
- 记忆层级提升
"""
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, Callable
import aiosqlite

from config.logging import get_logger


logger = get_logger(__name__)


class MemoryTaskScheduler:
    """记忆任务调度器"""

    def __init__(self, config: Any):
        """初始化调度器

        Args:
            config: 配置对象
        """
        self.config = config
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._get_manager_func: Optional[Callable] = None

    async def start(self, get_manager_func: Callable):
        """启动调度器

        Args:
            get_manager_func: 获取MemoryIndexManager的函数
        """
        if self._running:
            return

        self._running = True
        self._get_manager = get_manager_func

        # 启动后台任务
        self._task = asyncio.create_task(self._scheduler_loop())

        logger.info("[SCHEDULER] Started")

    async def stop(self):
        """停止调度器"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[SCHEDULER] Stopped")

    async def _scheduler_loop(self):
        """调度循环"""
        while self._running:
            try:
                # 每小时检查一次
                await asyncio.sleep(3600)

                # 检查是否到达执行时间（每天凌晨2点）
                now = datetime.now()
                if now.hour == 2 and now.minute < 10:
                    await self._run_daily_tasks()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[SCHEDULER] Error: {e}")

    async def _run_daily_tasks(self):
        """执行每日任务"""
        logger.info("[SCHEDULER] Running daily tasks")

        # 获取所有活跃用户
        user_ids = await self._get_active_users()

        for user_id in user_ids:
            try:
                manager = await self._get_manager(user_id)

                # 1. 清理工作记忆
                lifecycle_enabled = getattr(self.config, 'lifecycle', None)
                if lifecycle_enabled:
                    lifecycle_config = getattr(self.config.lifecycle, 'enabled', False)
                    if lifecycle_config and hasattr(manager, '_lifecycle_manager'):
                        await manager._lifecycle_manager.cleanup_working_memory()

                # 2. 提升记忆层级
                if lifecycle_enabled and lifecycle_config:
                    if hasattr(manager, '_lifecycle_manager'):
                        await manager._lifecycle_manager.promote_memories()

                # 3. 合并相似记忆
                consolidation_config = getattr(self.config, 'consolidation', None)
                if consolidation_config and getattr(consolidation_config, 'enabled', False):
                    from services.memory_consolidation import MemoryConsolidator

                    # 获取数据库连接
                    db_conn = manager._db

                    consolidator = MemoryConsolidator(
                        db_conn=db_conn,
                        user_id=user_id,
                        config=consolidation_config,
                    )
                    await consolidator.run_consolidation()

                logger.info(f"[SCHEDULER] Completed tasks for user {user_id}")

            except Exception as e:
                logger.error(f"[SCHEDULER] Error for user {user_id}: {e}")

    async def _get_active_users(self) -> list:
        """获取活跃用户列表"""
        # 从数据库获取有记忆数据的用户列表
        # 这里简化实现，从全局状态获取
        if self._get_manager_func:
            try:
                # 尝试从app.state获取
                import sys
                if 'api.main' in sys.modules:
                    from api.main import app
                    if hasattr(app.state, 'memory_managers'):
                        return list(app.state.memory_managers.keys())
            except Exception:
                pass

        return []

    async def run_now(self, user_ids: list = None):
        """立即运行一次任务（用于手动触发）

        Args:
            user_ids: 指定用户ID列表，None表示所有活跃用户
        """
        logger.info("[SCHEDULER] Running tasks on demand")

        if user_ids is None:
            user_ids = await self._get_active_users()

        for user_id in user_ids:
            try:
                manager = await self._get_manager(user_id)

                # 执行所有任务
                lifecycle_config = getattr(self.config.lifecycle, 'enabled', False) if hasattr(self.config, 'lifecycle') else False
                if lifecycle_config and hasattr(manager, '_lifecycle_manager'):
                    await manager._lifecycle_manager.cleanup_working_memory()
                    await manager._lifecycle_manager.promote_memories()

                consolidation_config = getattr(self.config, 'consolidation', None)
                if consolidation_config and getattr(consolidation_config, 'enabled', False):
                    from services.memory_consolidation import MemoryConsolidator

                    consolidator = MemoryConsolidator(
                        db_conn=manager._db,
                        user_id=user_id,
                        config=consolidation_config,
                    )
                    await consolidator.run_consolidation()

                logger.info(f"[SCHEDULER] On-demand tasks completed for user {user_id}")

            except Exception as e:
                logger.error(f"[SCHEDULER] Error for user {user_id}: {e}")


# 全局调度器实例
_scheduler: Optional[MemoryTaskScheduler] = None


def get_scheduler() -> Optional[MemoryTaskScheduler]:
    """获取全局调度器实例"""
    return _scheduler


def set_scheduler(scheduler: MemoryTaskScheduler):
    """设置全局调度器实例"""
    global _scheduler
    _scheduler = scheduler


async def stop_scheduler():
    """停止全局调度器"""
    global _scheduler
    if _scheduler:
        await _scheduler.stop()
        _scheduler = None
