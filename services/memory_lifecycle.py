"""记忆生命周期管理

实现三层记忆架构：
- 工作记忆 (Working Memory): 7天内，低重要性，定期清理
- 长期记忆 (Long-term Memory): 高重要性，完整保留
- 档案记忆 (Archive Memory): 超过90天，压缩存储
"""
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import aiosqlite

from .memory_importance import MemoryImportanceScorer, get_scorer, ImportanceScores
from .entity_extractor import EntityExtractor, get_entity_extractor, ExtractedEntities
from config.logging import get_logger


logger = get_logger(__name__)


class MemoryLifecycleManager:
    """记忆生命周期管理器"""

    def __init__(
        self,
        db_conn: aiosqlite.Connection,
        user_id: int,
        config: Any,
    ):
        """初始化生命周期管理器

        Args:
            db_conn: 数据库连接
            user_id: 用户ID
            config: 生命周期配置
        """
        self.db = db_conn
        self.user_id = user_id
        self.config = config

        self.scorer: Optional[MemoryImportanceScorer] = None
        self.entity_extractor: Optional[EntityExtractor] = None

    async def initialize(self, llm_forwarder=None):
        """初始化管理器"""
        # 获取重要性评分配置
        importance_config = getattr(self.config, 'importance', None)
        weights = None
        if importance_config and hasattr(importance_config, 'weights'):
            weights = {
                "novelty": importance_config.weights.novelty,
                "sentiment": importance_config.weights.sentiment,
                "feedback": importance_config.weights.feedback,
                "access": importance_config.weights.access,
                "density": importance_config.weights.density,
            }

        self.scorer = MemoryImportanceScorer(self.db, weights)
        self.entity_extractor = get_entity_extractor(llm_forwarder)

    async def process_new_chunk(
        self,
        chunk_id: str,
        chunk_text: str,
        embedding: list,
    ) -> str:
        """处理新记忆块：评分、提取、分层

        Args:
            chunk_id: 记忆块ID
            chunk_text: 记忆文本
            embedding: 嵌入向量

        Returns:
            分配的记忆层级
        """
        # 1. 计算重要性评分
        scores = await self.scorer.score_chunk(
            chunk_id=chunk_id,
            user_id=self.user_id,
            chunk_text=chunk_text,
            embedding=embedding,
        )

        # 2. 提取结构化信息
        entities: ExtractedEntities = await self.entity_extractor.extract(
            chunk_text,
            use_llm=scores.overall > 0.7,  # 高重要性才用LLM
        )

        # 3. 确定记忆层级
        tier = self._determine_tier(scores.overall)

        # 4. 保存元数据
        await self.scorer.save_scores(
            chunk_id=chunk_id,
            user_id=self.user_id,
            scores=scores,
            entities=entities.entities,
            key_points=entities.key_points,
        )

        logger.debug(
            f"[LIFECYCLE] Chunk {chunk_id[:8]}... → tier={tier}, "
            f"importance={scores.overall:.2f}"
        )

        return tier

    def _determine_tier(self, importance_score: float) -> str:
        """根据重要性分数确定记忆层级"""
        long_term_threshold = getattr(self.config, 'long_term_threshold', 0.6)
        archive_threshold = getattr(self.config, 'archive_threshold', 0.3)

        if importance_score >= long_term_threshold:
            return "long_term"
        elif importance_score >= archive_threshold:
            return "working"
        else:
            return "archive"

    async def cleanup_working_memory(self) -> Dict[str, int]:
        """清理过期的工作记忆

        Returns:
            清理统计
        """
        working_days = getattr(self.config, 'working_memory_days', 7)
        cutoff_date = datetime.now() - timedelta(days=working_days)

        # 删除过期的工作记忆（从元数据表标记为archive）
        cursor = await self.db.execute("""
            UPDATE memory_chunks_meta
            SET memory_tier = 'archive', updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?
              AND memory_tier = 'working'
              AND created_at < ?
        """, (self.user_id, cutoff_date.isoformat()))

        affected = cursor.rowcount
        await cursor.close()
        await self.db.commit()

        logger.info(f"[LIFECYCLE] Cleaned {affected} expired working memories")

        return {"cleaned": affected}

    async def get_tier_stats(self) -> Dict[str, int]:
        """获取各层级记忆数量统计

        Returns:
            {tier: count} 字典
        """
        cursor = await self.db.execute("""
            SELECT memory_tier, COUNT(*) as count
            FROM memory_chunks_meta
            WHERE user_id = ?
            GROUP BY memory_tier
        """, (self.user_id,))

        rows = await cursor.fetchall()
        await cursor.close()

        return {row[0]: row[1] for row in rows}

    async def promote_memories(self) -> Dict[str, int]:
        """提升记忆层级

        根据访问模式自动将工作记忆提升为长期记忆
        """
        promotion_count = getattr(self.config, 'promotion_access_count', 5)

        # 高访问次数的工作记忆 → 长期记忆
        cursor = await self.db.execute("""
            UPDATE memory_chunks_meta
            SET memory_tier = 'long_term', updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?
              AND memory_tier = 'working'
              AND access_count >= ?
        """, (self.user_id, promotion_count))

        promoted = cursor.rowcount
        await cursor.close()
        await self.db.commit()

        if promoted > 0:
            logger.info(f"[LIFECYCLE] Promoted {promoted} memories to long_term")

        return {"promoted": promoted}

    async def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息

        Returns:
            统计信息字典
        """
        # 获取层级统计
        tier_stats = await self.get_tier_stats()

        # 获取总体统计
        cursor = await self.db.execute("""
            SELECT
                COUNT(*) as total,
                AVG(importance_score) as avg_importance,
                SUM(access_count) as total_accesses
            FROM memory_chunks_meta
            WHERE user_id = ?
        """, (self.user_id,))

        row = await cursor.fetchone()
        await cursor.close()

        return {
            "user_id": self.user_id,
            "tiers": tier_stats,
            "total_chunks": row[0] if row else 0,
            "avg_importance": float(row[1]) if row and row[1] else 0.0,
            "total_accesses": row[2] if row else 0,
        }

    async def archive_old_memories(self, days: int = 90) -> Dict[str, int]:
        """归档旧记忆

        Args:
            days: 归档超过此天数的长期记忆

        Returns:
            归档统计
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        cursor = await self.db.execute("""
            UPDATE memory_chunks_meta
            SET memory_tier = 'archive', updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?
              AND memory_tier = 'long_term'
              AND created_at < ?
              AND access_count < 5
        """, (self.user_id, cutoff_date.isoformat()))

        affected = cursor.rowcount
        await cursor.close()
        await self.db.commit()

        logger.info(f"[LIFECYCLE] Archived {affected} old long-term memories")

        return {"archived": affected}
