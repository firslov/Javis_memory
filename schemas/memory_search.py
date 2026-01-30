"""记忆搜索相关的请求和响应模型

定义 RAG 向量检索记忆系统的搜索、同步、状态等模型。
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class MemorySearchRequest(BaseModel):
    """Request model for memory search."""

    query: str = Field(..., description="Search query")
    max_results: int = Field(6, ge=1, le=50, description="Maximum number of results")
    min_score: float = Field(0.35, ge=0.0, le=1.0, description="Minimum relevance score")


class MemorySearchResult(BaseModel):
    """A single memory search result."""

    chunk_id: str
    text: str
    path: str
    source: str
    start_line: int
    end_line: int
    vector_score: float
    keyword_score: float
    combined_score: float
    model: str
    provider: str
    token_count: int


class MemorySearchResponse(BaseModel):
    """Response model for memory search."""

    results: List[MemorySearchResult]
    query: str
    total_results: int


class MemorySyncResponse(BaseModel):
    """Response model for memory sync."""

    status: str
    files_scanned: int = 0
    files_updated: int = 0
    chunks_added: int = 0
    chunks_removed: int = 0
    errors: List[str] = Field(default_factory=list)


class MemoryStatusResponse(BaseModel):
    """Response model for memory status."""

    user_id: int
    db_path: str
    chunker_tokens: Optional[int] = None
    embedding_provider: Optional[str] = None
    embedding_dimension: Optional[int] = None
    syncing: bool = False
