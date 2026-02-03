"""记忆合并与去重服务

定期检测相似记忆簇，合并为更高层级的摘要
"""
import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import aiosqlite

from config.logging import get_logger


logger = get_logger(__name__)


class MemoryConsolidator:
    """记忆合并器"""

    def __init__(
        self,
        db_conn: aiosqlite.Connection,
        user_id: int,
        config: Any,
        llm_forwarder=None,
    ):
        """初始化合并器

        Args:
            db_conn: 数据库连接
            user_id: 用户ID
            config: 合并配置
            llm_forwarder: LLM转发器（用于生成摘要）
        """
        self.db = db_conn
        self.user_id = user_id
        self.config = config
        self.llm_forwarder = llm_forwarder

    async def find_similar_clusters(
        self,
        threshold: float = 0.85,
        time_window: timedelta = timedelta(days=7),
        min_cluster_size: int = 3,
    ) -> List[List[Dict]]:
        """检测相似记忆簇

        Args:
            threshold: 相似度阈值
            time_window: 时间窗口
            min_cluster_size: 最小簇大小

        Returns:
            记忆簇列表，每个簇是一组相似的记忆
        """
        cutoff_date = datetime.now() - time_window

        # 获取候选记忆 - 只处理未合并的记忆
        cursor = await self.db.execute("""
            SELECT c.id, c.text, m.importance_score, m.created_at
            FROM memory_chunks c
            INNER JOIN memory_chunks_meta m ON c.id = m.chunk_id AND c.user_id = m.user_id
            WHERE c.user_id = ?
              AND m.created_at > ?
              AND m.memory_tier != 'consolidated'
            ORDER BY c.created_at DESC
        """, (self.user_id, cutoff_date.isoformat()))

        rows = await cursor.fetchall()
        await cursor.close()

        # 使用向量相似度进行聚类
        clusters = []
        processed = set()

        for row in rows:
            chunk_id, text, importance, created_at = row

            if chunk_id in processed:
                continue

            # 找到与当前记忆相似的其他记忆
            similar_chunks = await self._find_similar_chunks(
                chunk_id, rows, processed, threshold
            )

            # 如果簇足够大，添加到结果
            if len(similar_chunks) + 1 >= min_cluster_size:
                clusters.append([
                    dict(chunk_id=chunk_id, text=text, importance=importance or 0.5),
                    *similar_chunks,
                ])
                processed.add(chunk_id)

        logger.info(f"[CONSOLIDATE] Found {len(clusters)} clusters")
        return clusters

    async def _find_similar_chunks(
        self,
        chunk_id: str,
        all_rows: List,
        processed: set,
        threshold: float,
    ) -> List[Dict]:
        """找到与给定chunk相似的其他chunk

        使用向量相似度进行高效查找
        """
        similar = []

        # 获取当前chunk的向量
        cursor = await self.db.execute("""
            SELECT embedding FROM memory_chunks_vec WHERE id = ? AND user_id = ?
        """, (chunk_id, self.user_id))

        row = await cursor.fetchone()
        await cursor.close()

        if not row:
            return similar

        import sqlite_vec
        current_vec = sqlite_vec.deserialize_float32(row[0])

        # 计算与其他chunk的相似度
        for other_row in all_rows:
            other_id = other_row[0]
            if other_id in processed or other_id == chunk_id:
                continue

            # 获取其他chunk的向量
            cursor = await self.db.execute("""
                SELECT embedding FROM memory_chunks_vec WHERE id = ? AND user_id = ?
            """, (other_id, self.user_id))

            vec_row = await cursor.fetchone()
            await cursor.close()

            if not vec_row:
                continue

            other_vec = sqlite_vec.deserialize_float32(vec_row[0])

            # 计算余弦相似度（向量已归一化）
            import math
            dot_product = sum(v1 * v2 for v1, v2 in zip(current_vec, other_vec))
            similarity = max(0.0, min(1.0, dot_product))

            if similarity >= threshold:
                similar.append(dict(
                    chunk_id=other_id,
                    text=other_row[1],
                    importance=other_row[2] or 0.5,
                ))
                processed.add(other_id)

        return similar

    async def consolidate_cluster(
        self,
        cluster: List[Dict],
    ) -> Optional[Dict]:
        """合并一个记忆簇

        Args:
            cluster: 相似记忆列表

        Returns:
            合并后的记忆元数据
        """
        if not cluster:
            return None

        # 选择最高重要性的记忆作为主要记忆
        primary = max(cluster, key=lambda x: x.get("importance", 0) or 0)

        # 生成合并摘要
        summary = await self._generate_consolidated_summary(cluster)

        # 更新主要记忆的元数据
        chunk_id = primary["chunk_id"]

        # 获取现有元数据
        cursor = await self.db.execute("""
            SELECT entities, key_points FROM memory_chunks_meta
            WHERE chunk_id = ? AND user_id = ?
        """, (chunk_id, self.user_id))

        row = await cursor.fetchone()
        await cursor.close()

        entities = json.loads(row[0]) if row and row[0] else []
        key_points = json.loads(row[1]) if row and row[1] else []

        # 添加合并标记
        key_points.append(f"[Consolidated from {len(cluster)} similar memories]")
        key_points.append(f"Summary: {summary}")

        # 更新元数据
        await self.db.execute("""
            UPDATE memory_chunks_meta
            SET key_points = ?, updated_at = CURRENT_TIMESTAMP
            WHERE chunk_id = ? AND user_id = ?
        """, (json.dumps(key_points), chunk_id, self.user_id))

        # 标记其他记忆为已合并
        for member in cluster:
            if member["chunk_id"] != chunk_id:
                await self.db.execute("""
                    UPDATE memory_chunks_meta
                    SET memory_tier = 'consolidated', updated_at = CURRENT_TIMESTAMP
                    WHERE chunk_id = ? AND user_id = ?
                """, (member["chunk_id"], self.user_id))

        await self.db.commit()

        logger.info(f"[CONSOLIDATE] Consolidated cluster around {chunk_id[:8]}...")
        return {"chunk_id": chunk_id, "summary": summary}

    async def _generate_consolidated_summary(
        self,
        cluster: List[Dict],
    ) -> str:
        """生成簇的合并摘要

        Args:
            cluster: 相似记忆列表

        Returns:
            合并摘要文本
        """
        # 合并所有文本
        all_text = "\n".join(m["text"] for m in cluster)

        # 简单摘要：取前200字符
        if len(all_text) <= 200:
            return all_text

        # 使用LLM生成摘要（如果可用）
        if self.llm_forwarder:
            try:
                models = self.llm_forwarder.get_available_models()
                if models:
                    model = list(models.keys())[0]

                    prompt = f"""请将以下{len(cluster)}条相似记忆合并为一个简洁的摘要（不超过200字）：

{all_text[:2000]}

摘要："""

                    from openai import AsyncOpenAI
                    server = list(self.llm_forwarder.servers.values())[0]
                    client = AsyncOpenAI(
                        base_url=server.base_url,
                        api_key=server.api_key,
                    )

                    response = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                    )

                    summary = response.choices[0].message.content

                    if summary:
                        return summary[:200]

            except Exception as e:
                logger.warning(f"[CONSOLIDATE] LLM summary failed: {e}")

        # 回退：简单截取
        return all_text[:200] + "..."

    async def run_consolidation(
        self,
        threshold: float = None,
        time_window: timedelta = None,
    ) -> Dict[str, Any]:
        """执行完整的合并流程

        Returns:
            合并统计
        """
        threshold = threshold or getattr(self.config, 'similarity_threshold', 0.85)
        time_window_days = getattr(self.config, 'time_window_days', 7)
        time_window = time_window or timedelta(days=time_window_days)
        min_cluster_size = getattr(self.config, 'min_cluster_size', 3)

        logger.info(f"[CONSOLIDATE] Starting consolidation: threshold={threshold}")

        # 1. 检测相似簇
        clusters = await self.find_similar_clusters(
            threshold=threshold,
            time_window=time_window,
            min_cluster_size=min_cluster_size,
        )

        # 2. 合并每个簇
        consolidated = 0
        for cluster in clusters:
            result = await self.consolidate_cluster(cluster)
            if result:
                consolidated += 1

        stats = {
            "clusters_found": len(clusters),
            "clusters_consolidated": consolidated,
            "memories_consolidated": sum(len(c) for c in clusters),
        }

        logger.info(f"[CONSOLIDATE] Completed: {stats}")
        return stats
