"""RAG 记忆系统的数据库模型

定义向量检索记忆系统的数据表：
    - MemoryFile: 源文件元数据
    - MemoryChunk: 文本块和嵌入向量
    - EmbeddingCache: 嵌入结果缓存
    - MemoryIndexMeta: 索引配置元数据

此模块替代了原有的 4 层记忆架构。
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Text, Integer, Float, JSON, DateTime, Index, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class MemoryBase(DeclarativeBase):
    """Base class for memory models."""
    pass


# ============================================================================
# Memory source files
# ============================================================================

class MemoryFile(MemoryBase):
    """Source file metadata for memory indexing.

    Tracks which files have been indexed and their current state.
    """
    __tablename__ = "memory_files"

    # Composite primary key: user_id + path
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    path: Mapped[str] = mapped_column(String(500), primary_key=True)

    # Source type: "memory", "sessions", "custom"
    source: Mapped[str] = mapped_column(String(50), default="memory")

    # File state
    hash: Mapped[str] = mapped_column(String(64))  # SHA-256
    mtime: Mapped[int] = mapped_column(Integer)  # File modification time
    size: Mapped[int] = mapped_column(Integer)  # File size in bytes

    # Indexing state
    indexed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ============================================================================
# Text chunks with embeddings
# ============================================================================

class MemoryChunk(MemoryBase):
    """Text chunk with embedding vector.

    Stores individual chunks of text with their embeddings for similarity search.
    """
    __tablename__ = "memory_chunks"

    # Unique chunk ID
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # SHA-256 hash

    # User association
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Source information
    path: Mapped[str] = mapped_column(String(500))
    source: Mapped[str] = mapped_column(String(50), default="memory")

    # Line numbers
    start_line: Mapped[int] = mapped_column(Integer)
    end_line: Mapped[int] = mapped_column(Integer)

    # Content
    hash: Mapped[str] = mapped_column(String(64))  # Content hash
    text: Mapped[str] = mapped_column(Text)

    # Embedding
    model: Mapped[str] = mapped_column(String(100))  # Model used
    provider: Mapped[str] = mapped_column(String(50))  # Provider: openai, gemini, local
    dimension: Mapped[int] = mapped_column(Integer)  # Vector dimension
    embedding: Mapped[str] = mapped_column(Text)  # JSON-encoded vector

    # Metadata
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    extra_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Indexes
    __table_args__ = (
        Index("ix_memory_chunks_user_path", "user_id", "path"),
        Index("ix_memory_chunks_source", "user_id", "source"),
        Index("ix_memory_chunks_model", "model", "provider"),
    )


# ============================================================================
# Embedding cache
# ============================================================================

class EmbeddingCache(MemoryBase):
    """Cache for embedding results.

    Avoids re-computing embeddings for identical text chunks.
    """
    __tablename__ = "embedding_cache"

    # Composite key
    provider: Mapped[str] = mapped_column(String(50), primary_key=True)
    model: Mapped[str] = mapped_column(String(100), primary_key=True)
    hash: Mapped[str] = mapped_column(String(64), primary_key=True)

    # Cached embedding
    dimension: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[str] = mapped_column(Text)  # JSON-encoded vector

    # Statistics
    hit_count: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamp
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ============================================================================
# Memory index metadata
# ============================================================================

class MemoryIndexMeta(MemoryBase):
    """Metadata for the memory index.

    Tracks configuration to detect when full reindex is needed.
    Based on AI_MEMORY_SYSTEM_DESIGN.md recommendations.
    """
    __tablename__ = "memory_index_meta"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(Text)


# ============================================================================
# Memory chunk metadata (importance scoring, lifecycle management)
# ============================================================================

class MemoryChunkMeta(MemoryBase):
    """记忆块元数据 - 用于重要性评估和生命周期管理.

    Stores:
    - Importance scores (overall and per-dimension)
    - Access statistics
    - Memory tier (working/long_term/archive)
    - Extracted entities and key points
    """
    __tablename__ = "memory_chunks_meta"

    # 复合主键
    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # 重要性评分
    importance_score: Mapped[float] = mapped_column(Float, default=0.5)
    novelty_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feedback_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # 访问统计
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # 记忆层级
    memory_tier: Mapped[str] = mapped_column(String(20), default="working")  # working/long_term/archive

    # 提取的结构化信息
    entities: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    key_points: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    # 时间戳
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 索引
    __table_args__ = (
        Index("ix_memory_meta_tier", "user_id", "memory_tier"),
        Index("ix_memory_meta_importance", "user_id", "importance_score"),
        Index("ix_memory_meta_access", "user_id", "access_count", "last_accessed_at"),
    )


class UserMemoryProfile(MemoryBase):
    """用户记忆偏好画像.

    Stores user preferences and statistics for memory system.
    """
    __tablename__ = "user_memory_profiles"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # 偏好设置
    preferred_result_size: Mapped[int] = mapped_column(Integer, default=6)
    precision_preference: Mapped[float] = mapped_column(Float, default=0.5)

    # 统计信息
    total_interactions: Mapped[int] = mapped_column(Integer, default=0)
    feedback_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ============================================================================
# Helper functions
# ============================================================================

META_KEY = "memory_index_meta_v1"
