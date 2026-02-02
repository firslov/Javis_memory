"""记忆重要性评分服务

多维度评估记忆价值：
- 新颖性 (Novelty): 与现有记忆的相似度
- 情感强度 (Sentiment): 文本情感分析得分
- 用户反馈 (Feedback): 用户显式/隐式反馈
- 访问频率 (Access): 被检索的次数
- 信息密度 (Density): 实体数量、结构化信息占比
"""
import asyncio
import json
import math
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiosqlite

from config.logging import get_logger


logger = get_logger(__name__)


@dataclass
class ImportanceScores:
    """重要性评分结果"""
    overall: float  # 综合评分 0-1
    novelty: float  # 新颖性 0-1
    sentiment: float  # 情感强度 0-1
    feedback: float  # 用户反馈 0-1
    access: float  # 访问频率 0-1
    density: float  # 信息密度 0-1


class MemoryImportanceScorer:
    """记忆重要性评分器"""

    def __init__(
        self,
        db_conn: aiosqlite.Connection,
        weights: Optional[Dict[str, float]] = None,
    ):
        """初始化评分器

        Args:
            db_conn: 数据库连接
            weights: 各维度权重，默认为均衡权重
        """
        self.db = db_conn
        self.weights = weights or {
            "novelty": 0.25,
            "sentiment": 0.15,
            "feedback": 0.30,
            "access": 0.20,
            "density": 0.10,
        }

    async def score_chunk(
        self,
        chunk_id: str,
        user_id: int,
        chunk_text: str,
        embedding: list,
    ) -> ImportanceScores:
        """对单个记忆块进行评分

        Args:
            chunk_id: 记忆块ID
            user_id: 用户ID
            chunk_text: 记忆文本
            embedding: 嵌入向量

        Returns:
            ImportanceScores 评分结果
        """
        # 并行计算各维度分数
        novelty, sentiment, density = await asyncio.gather(
            self._calculate_novelty(embedding, user_id),
            self._calculate_sentiment(chunk_text),
            self._calculate_density(chunk_text),
        )

        # 获取历史数据
        feedback, access = await self._get_historical_scores(chunk_id, user_id)

        # 加权计算综合评分
        overall = (
            self.weights["novelty"] * novelty +
            self.weights["sentiment"] * sentiment +
            self.weights["feedback"] * feedback +
            self.weights["access"] * access +
            self.weights["density"] * density
        )

        return ImportanceScores(
            overall=overall,
            novelty=novelty,
            sentiment=sentiment,
            feedback=feedback,
            access=access,
            density=density,
        )

    async def _calculate_novelty(
        self,
        embedding: list,
        user_id: int,
    ) -> float:
        """计算新颖性分数

        新颖性 = 1 - 与现有记忆的最大相似度
        """
        try:
            # 归一化查询向量
            norm = math.sqrt(sum(v * v for v in embedding))
            if norm > 0:
                embedding = [v / norm for v in embedding]

            query_json = json.dumps(embedding)

            # 查询最相似的5个记忆
            cursor = await self.db.execute("""
                SELECT distance
                FROM memory_chunks_vec
                WHERE embedding MATCH ?
                  AND user_id = ?
                ORDER BY distance
                LIMIT 5
            """, (query_json, user_id))

            rows = await cursor.fetchall()
            await cursor.close()

            if not rows:
                return 1.0  # 没有历史记忆，完全新颖

            # 获取最大相似度
            max_distance = rows[0][0]
            max_similarity = 1.0 - (max_distance * max_distance / 2.0)

            # 新颖性 = 1 - 最大相似度
            return max(0.0, 1.0 - max_similarity)

        except Exception as e:
            # 向量搜索不可用时，返回中等新颖性分数
            logger.debug(f"[SCORER] Novelty calculation failed, using default: {e}")
            return 0.5  # 返回中等新颖性

    async def _calculate_sentiment(self, text: str) -> float:
        """计算情感强度分数

        使用简单的关键词匹配，可升级为调用情感分析API
        """
        # 强情感关键词
        strong_keywords = [
            "非常重要", "关键", "紧急", "必须", "务必",
            "喜欢", "讨厌", "爱", "恨",
            "very important", "critical", "urgent", "must",
            "love", "hate", "favorite", "essential",
        ]

        # 中等情感关键词
        medium_keywords = [
            "需要", "应该", "希望", "想要",
            "need", "should", "want", "hope", "prefer",
        ]

        text_lower = text.lower()
        strong_count = sum(1 for kw in strong_keywords if kw in text_lower)
        medium_count = sum(1 for kw in medium_keywords if kw in text_lower)

        # 计算分数 (0-1)
        score = min(1.0, (strong_count * 0.3 + medium_count * 0.15))
        return score

    async def _calculate_density(self, text: str) -> float:
        """计算信息密度分数

        基于实体数量、文本长度、结构化程度
        """
        import re

        # 检测实体模式
        patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "url": r'https?://\S+',
            "number": r'\b\d+\.?\d*\b',
            "hashtag": r'#[A-Za-z0-9_]+',
            "mention": r'@[A-Za-z0-9_]+',
        }

        entity_count = 0
        for pattern in patterns.values():
            entity_count += len(re.findall(pattern, text))

        # 代码块密度
        code_blocks = text.count("```")

        # 标题密度
        headers = len(re.findall(r'^#{1,6}\s', text, re.MULTILINE))

        # 计算密度分数
        text_length = len(text)
        if text_length == 0:
            return 0.0

        base_score = min(1.0, entity_count / 10.0)  # 每10个实体为1分
        code_bonus = min(0.3, code_blocks * 0.1)
        header_bonus = min(0.2, headers * 0.05)

        return min(1.0, base_score + code_bonus + header_bonus)

    async def _get_historical_scores(
        self,
        chunk_id: str,
        user_id: int,
    ) -> tuple[float, float]:
        """获取历史反馈和访问分数

        Returns:
            (feedback_score, access_score)
        """
        cursor = await self.db.execute("""
            SELECT access_count, last_accessed_at
            FROM memory_chunks_meta
            WHERE chunk_id = ? AND user_id = ?
        """, (chunk_id, user_id))

        row = await cursor.fetchone()
        await cursor.close()

        if not row:
            return (0.0, 0.0)

        access_count, last_accessed = row

        # 访问分数：基于访问次数和时间衰减
        access_score = min(1.0, (access_count or 0) / 10.0)  # 10次访问为满分

        # 时间衰减：最近访问的权重更高
        if last_accessed:
            try:
                accessed_at = datetime.fromisoformat(last_accessed)
                days_since = (datetime.now() - accessed_at).days
                decay = 0.95 ** max(0, days_since)  # 每天衰减5%
                access_score *= decay
            except (ValueError, TypeError):
                pass

        # 反馈分数：暂时基于访问分数（后续可扩展显式反馈）
        feedback_score = access_score

        return (feedback_score, access_score)

    async def save_scores(
        self,
        chunk_id: str,
        user_id: int,
        scores: ImportanceScores,
        entities: Optional[list] = None,
        key_points: Optional[list] = None,
    ) -> None:
        """保存评分到数据库

        Args:
            chunk_id: 记忆块ID
            user_id: 用户ID
            scores: 评分结果
            entities: 提取的实体列表
            key_points: 提取的关键点
        """
        tier = self._determine_tier(scores.overall)

        await self.db.execute("""
            INSERT OR REPLACE INTO memory_chunks_meta
            (chunk_id, user_id, importance_score, novelty_score, sentiment_score,
             feedback_score, access_count, last_accessed_at, memory_tier,
             entities, key_points, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT access_count FROM memory_chunks_meta WHERE chunk_id = ? AND user_id = ?), 0),
                COALESCE((SELECT last_accessed_at FROM memory_chunks_meta WHERE chunk_id = ? AND user_id = ?), NULL),
                ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            chunk_id,
            user_id,
            scores.overall,
            scores.novelty,
            scores.sentiment,
            scores.feedback,
            chunk_id, user_id,  # For subquery
            chunk_id, user_id,  # For subquery
            tier,
            json.dumps(entities) if entities else None,
            json.dumps(key_points) if key_points else None,
        ))

        await self.db.commit()

    def _determine_tier(self, importance_score: float) -> str:
        """根据重要性分数确定记忆层级

        Args:
            importance_score: 综合重要性分数 (0-1)

        Returns:
            层级: 'working', 'long_term', 'archive'
        """
        if importance_score >= 0.7:
            return "long_term"
        elif importance_score >= 0.3:
            return "working"
        else:
            return "archive"

    async def update_access(
        self,
        chunk_id: str,
        user_id: int,
    ) -> None:
        """更新访问统计

        Args:
            chunk_id: 记忆块ID
            user_id: 用户ID
        """
        await self.db.execute("""
            INSERT INTO memory_chunks_meta (chunk_id, user_id, access_count, last_accessed_at, memory_tier, updated_at)
            VALUES (?, ?, 1, CURRENT_TIMESTAMP, 'working', CURRENT_TIMESTAMP)
            ON CONFLICT(chunk_id, user_id) DO UPDATE SET
                access_count = access_count + 1,
                last_accessed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
        """, (chunk_id, user_id))

        await self.db.commit()


# 便捷函数：获取或创建评分器
_scorers: Dict[int, MemoryImportanceScorer] = {}


def get_scorer(
    user_id: int,
    db_conn: aiosqlite.Connection,
    weights: Optional[Dict[str, float]] = None,
) -> MemoryImportanceScorer:
    """获取或创建评分器实例"""
    if user_id not in _scorers:
        _scorers[user_id] = MemoryImportanceScorer(db_conn, weights)
    return _scorers[user_id]


def clear_scorers():
    """清除所有评分器实例（用于测试或重置）"""
    _scorers.clear()
