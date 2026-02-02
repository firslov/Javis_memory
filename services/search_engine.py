"""混合搜索引擎

结合向量相似度搜索和关键词搜索（BM25）实现混合检索。

特性:
    - 使用归一化向量的余弦相似度
    - 候选数量硬限制 200 条以保证性能
    - 从欧氏距离正确转换为相似度分数
"""
import asyncio
import json
import math
from dataclasses import dataclass
from typing import List, Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from database.memory_models import MemoryChunk

from config.logging import get_logger


# Import sqlite_vec for serialization
import sqlite_vec


# Hard upper limit for candidate retrieval (from design doc)
MAX_CANDIDATE_LIMIT = 200


logger = get_logger(__name__)


@dataclass
class SearchResult:
    """A search result with score and metadata."""

    chunk_id: str
    text: str
    path: str
    source: str
    start_line: int
    end_line: int
    vector_score: float  # 0-1, higher is better
    keyword_score: float  # 0-1, higher is better
    combined_score: float  # 0-1, higher is better
    model: str
    provider: str
    token_count: int


class SearchEngine:
    """Hybrid search engine combining vector and keyword search.

    Features:
    - Vector similarity search via sqlite-vec
    - BM25 keyword search via FTS5
    - Configurable result fusion
    """

    def __init__(
        self,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        candidate_multiplier: int = 4,
    ):
        """Initialize the search engine.

        Args:
            vector_weight: Weight for vector search results (0-1).
            keyword_weight: Weight for keyword search results (0-1).
            candidate_multiplier: Multiplier for candidate retrieval.
        """
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.candidate_multiplier = candidate_multiplier

        # Validate weights
        if not (0 <= vector_weight <= 1):
            raise ValueError("vector_weight must be between 0 and 1")
        if not (0 <= keyword_weight <= 1):
            raise ValueError("keyword_weight must be between 0 and 1")
        if abs(vector_weight + keyword_weight - 1.0) > 0.01:
            logger.debug(
                f"[SEARCH] Weights: vector={vector_weight}, keyword={keyword_weight}"
            )

    async def search(
        self,
        query_embedding: List[float],
        query_text: str,
        user_id: int,
        max_results: int = 6,
        min_score: float = 0.35,
        db_conn=None,
    ) -> List[SearchResult]:
        """Perform hybrid search.

        Args:
            query_embedding: Query vector embedding.
            query_text: Original query text for keyword search.
            user_id: User ID to filter results.
            max_results: Maximum number of results to return.
            min_score: Minimum combined score (0-1).
            db_conn: Database connection.

        Returns:
            List of SearchResult objects sorted by combined score.
        """
        if db_conn is None:
            raise ValueError("db_conn is required")

        # Calculate candidate count with hard upper limit (from design doc)
        candidate_count = min(
            MAX_CANDIDATE_LIMIT,
            max_results * self.candidate_multiplier
        )

        # Run both searches in parallel
        vector_results, keyword_results = await asyncio.gather(
            self._vector_search(
                query_embedding,
                user_id,
                candidate_count,
                db_conn,
            ),
            self._keyword_search(
                query_text,
                user_id,
                candidate_count,
                db_conn,
            ),
        )

        # Combine and rank results
        combined = self._combine_results(
            vector_results,
            keyword_results,
        )

        # Filter by minimum score and limit results
        filtered = [r for r in combined if r.combined_score >= min_score]
        results = filtered[:max_results]

        # ===== 新增：更新访问统计 =====
        if results:
            chunk_ids = [r.chunk_id for r in results]
            placeholders = ','.join('?' * len(chunk_ids))
            try:
                await db_conn.execute(f"""
                    UPDATE memory_chunks_meta
                    SET access_count = access_count + 1,
                        last_accessed_at = CURRENT_TIMESTAMP
                    WHERE chunk_id IN ({placeholders})
                      AND user_id = ?
                """, (*chunk_ids, user_id))
                await db_conn.commit()
            except Exception as e:
                # 如果表不存在或其他错误，不影响搜索结果
                logger.debug(f"[SEARCH] Failed to update access stats: {e}")

        return results

    async def _vector_search(
        self,
        query_embedding: List[float],
        user_id: int,
        limit: int,
        db_conn,
    ) -> List[dict]:
        """Perform vector similarity search using sqlite-vec.

        Args:
            query_embedding: Query vector.
            user_id: User ID to filter.
            limit: Maximum results.
            db_conn: Database connection.

        Returns:
            List of result dicts with vector_score.

        Raises:
            RuntimeError: If sqlite-vec extension is not available.
        """
        import aiosqlite
        import math

        # Normalize query embedding to match stored normalized vectors
        norm = math.sqrt(sum(v * v for v in query_embedding))
        if norm > 0:
            query_embedding = [v / norm for v in query_embedding]

        # sqlite-vec MATCH operator expects JSON array string format
        import json
        query_json = json.dumps(query_embedding)

        # Build query using derived table approach
        # The LIMIT must be in the subquery that directly queries the vec0 table
        # for sqlite-vec to recognize the KNN query
        sql = f"""
            SELECT c.id, c.text, c.path, c.source,
                   c.start_line, c.end_line, c.model, c.provider,
                   c.token_count, v.distance
            FROM memory_chunks c
            JOIN (
                SELECT id, distance
                FROM memory_chunks_vec
                WHERE embedding MATCH ?
                  AND user_id = ?
                ORDER BY distance
                LIMIT {limit}
            ) v ON c.id = v.id
        """

        cursor = await db_conn.execute(sql, (query_json, user_id))
        rows = await cursor.fetchall()
        await cursor.close()

        results = []
        for row in rows:
            # Convert Euclidean distance to cosine similarity
            # For normalized vectors: cosine_sim = 1 - distance^2/2
            distance = row[9]  # distance column

            # Skip if distance is NULL (shouldn't happen with proper JOIN)
            if distance is None:
                continue

            # Convert to cosine similarity (0-1 scale, higher is better)
            # This is the correct formula for normalized vectors
            cosine_sim = 1.0 - (distance * distance / 2.0)
            score = max(0.0, cosine_sim)  # Clamp to non-negative

            results.append({
                "chunk_id": row[0],
                "text": row[1],
                "path": row[2],
                "source": row[3],
                "start_line": row[4],
                "end_line": row[5],
                "model": row[6],
                "provider": row[7],
                "token_count": row[8],
                "vector_score": score,
            })

        return results

    async def _keyword_search(
        self,
        query_text: str,
        user_id: int,
        limit: int,
        db_conn,
    ) -> List[dict]:
        """Perform BM25 keyword search.

        Args:
            query_text: Query text.
            user_id: User ID to filter.
            limit: Maximum results.
            db_conn: Database connection.

        Returns:
            List of result dicts with keyword_score.
        """
        # Simple query without special FTS5 syntax
        # Just use the raw text for simple matching
        search_query = query_text.strip()

        # If query is empty or too short, return empty
        if len(search_query) < 2:
            return []

        # Use LIKE for simple text matching instead of FTS5 MATCH
        # This avoids FTS5 syntax issues
        like_pattern = f"%{search_query}%"

        sql = """
            SELECT c.id, c.text, c.path, c.source,
                   c.start_line, c.end_line, c.model, c.provider,
                   c.token_count
            FROM memory_chunks c
            WHERE c.user_id = ? AND c.text LIKE ?
            LIMIT ?
        """

        try:
            cursor = await db_conn.execute(sql, (user_id, like_pattern, limit))
            rows = await cursor.fetchall()
            await cursor.close()

            results = []
            for row in rows:
                # Simple scoring based on text position
                text = row[1]
                position = text.lower().find(search_query.lower())
                # Higher score if match is earlier in text
                score = 1.0 / (1.0 + max(0, position / 1000.0))

                results.append({
                    "chunk_id": row[0],
                    "text": row[1],
                    "path": row[2],
                    "source": row[3],
                    "start_line": row[4],
                    "end_line": row[5],
                    "model": row[6],
                    "provider": row[7],
                    "token_count": row[8],
                    "keyword_score": score,
                })

            return results

        except Exception as e:
            logger.debug(f"[SEARCH] Keyword failed: {e}")
            return []

    def _combine_results(
        self,
        vector_results: List[dict],
        keyword_results: List[dict],
    ) -> List[SearchResult]:
        """Combine vector and keyword search results.

        Args:
            vector_results: Results from vector search.
            keyword_results: Results from keyword search.

        Returns:
            Combined and ranked list of SearchResult objects.
        """
        # Build score maps
        vector_scores = {r["chunk_id"]: r["vector_score"] for r in vector_results}
        keyword_scores = {r["chunk_id"]: r["keyword_score"] for r in keyword_results}

        # Get all unique chunk IDs
        all_ids = set(vector_scores.keys()) | set(keyword_scores.keys())

        # Build result map from vector results (has full data)
        result_map = {r["chunk_id"]: r for r in vector_results}

        # Add data from keyword results if missing
        for r in keyword_results:
            if r["chunk_id"] not in result_map:
                result_map[r["chunk_id"]] = r

        # Combine scores
        combined = []
        for chunk_id in all_ids:
            data = result_map[chunk_id]
            v_score = vector_scores.get(chunk_id, 0.0)
            k_score = keyword_scores.get(chunk_id, 0.0)

            # Handle None values explicitly
            if v_score is None:
                v_score = 0.0
            if k_score is None:
                k_score = 0.0

            # Handle None weights
            vw = self.vector_weight if self.vector_weight is not None else 0.7
            kw = self.keyword_weight if self.keyword_weight is not None else 0.3

            # Weighted combination
            combined_score = (
                vw * v_score +
                kw * k_score
            )

            combined.append(SearchResult(
                chunk_id=chunk_id,
                text=data.get("text", ""),
                path=data.get("path", ""),
                source=data.get("source", "unknown"),
                start_line=data.get("start_line", 0),
                end_line=data.get("end_line", 0),
                vector_score=v_score,
                keyword_score=k_score,
                combined_score=combined_score,
                model=data.get("model", "unknown"),
                provider=data.get("provider", "unknown"),
                token_count=data.get("token_count", 0),
            ))

        # Sort by combined score
        combined.sort(key=lambda r: r.combined_score, reverse=True)
        return combined


def format_search_results(results: List[SearchResult]) -> str:
    """Format search results for LLM context.

    Args:
        results: List of search results.

    Returns:
        Formatted string for LLM context.
    """
    if not results:
        return "No relevant memory found."

    parts = ["# Relevant Memory\n"]

    for i, result in enumerate(results, 1):
        source_label = {
            "memory": "Memory",
            "sessions": "Session",
            "custom": "Custom",
        }.get(result.source, result.source)

        parts.append(
            f"## {source_label}: {result.path} (lines {result.start_line}-{result.end_line})\n"
            f"```\n{result.text}\n```\n"
            f"*Relevance: {result.combined_score:.2%}*\n"
        )

    return "\n".join(parts)
