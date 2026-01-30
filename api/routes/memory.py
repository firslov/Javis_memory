"""记忆管理 API 路由

提供 RAG 向量检索记忆系统的搜索、同步和状态查询接口。
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Request
from config.settings import get_settings
from services.memory_index import MemoryIndexManager
from api.routes.dependencies import get_or_create_memory_manager

from schemas.memory_search import (
    MemorySearchRequest,
    MemorySearchResponse,
    MemorySearchResult,
    MemorySyncResponse,
    MemoryStatusResponse,
)


router = APIRouter(prefix="/v1/memory", tags=["memory"])


# ============================================================================
# RAG-based Memory Search API
# ============================================================================

@router.post("/search", response_model=MemorySearchResponse)
async def search_memory(
    request: MemorySearchRequest,
    user_id: int = Query(1, description="User ID"),
    http_request: Request = Request,
):
    """搜索记忆 (RAG 向量检索)

    使用向量相似度和 BM25 关键词搜索的组合检索相关记忆。
    """
    settings = get_settings()

    if not settings.memory_search.enabled:
        raise HTTPException(status_code=503, detail="Memory search is disabled")

    manager: MemoryIndexManager = await get_or_create_memory_manager(http_request, user_id)

    try:
        results = await manager.search(
            query=request.query,
            max_results=request.max_results,
            min_score=request.min_score,
        )

        return MemorySearchResponse(
            results=[
                MemorySearchResult(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    path=r.path,
                    source=r.source,
                    start_line=r.start_line,
                    end_line=r.end_line,
                    vector_score=r.vector_score,
                    keyword_score=r.keyword_score,
                    combined_score=r.combined_score,
                    model=r.model,
                    provider=r.provider,
                    token_count=r.token_count,
                )
                for r in results
            ],
            query=request.query,
            total_results=len(results),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/file")
async def read_memory_file(
    path: str = Query(..., description="Relative file path"),
    user_id: int = Query(1, description="User ID"),
    from_line: Optional[int] = Query(None, ge=1, description="Starting line number"),
    lines: Optional[int] = Query(None, ge=1, description="Number of lines to read"),
    http_request: Request = Request,
):
    """读取记忆文件内容"""
    settings = get_settings()

    if not settings.memory_search.enabled:
        raise HTTPException(status_code=503, detail="Memory search is disabled")

    manager: MemoryIndexManager = await get_or_create_memory_manager(http_request, user_id)

    try:
        content = await manager.read_file(path, from_line, lines)
        return {
            "path": path,
            "content": content,
            "from_line": from_line,
            "lines": lines,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Read failed: {str(e)}")


@router.post("/sync", response_model=MemorySyncResponse)
async def sync_memory(
    user_id: int = Query(1, description="User ID"),
    force: bool = Query(False, description="Force full re-index"),
    http_request: Request = Request,
):
    """同步记忆索引

    扫描记忆文件并更新向量索引。
    """
    settings = get_settings()

    if not settings.memory_search.enabled:
        raise HTTPException(status_code=503, detail="Memory search is disabled")

    manager: MemoryIndexManager = await get_or_create_memory_manager(http_request, user_id)

    try:
        stats = await manager.sync(force=force)
        return MemorySyncResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")


@router.get("/status", response_model=MemoryStatusResponse)
async def get_memory_status(
    user_id: int = Query(1, description="User ID"),
    http_request: Request = Request,
):
    """获取记忆系统状态"""
    settings = get_settings()

    if not settings.memory_search.enabled:
        raise HTTPException(status_code=503, detail="Memory search is disabled")

    manager: MemoryIndexManager = await get_or_create_memory_manager(http_request, user_id)

    try:
        status = manager.status()
        return MemoryStatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status failed: {str(e)}")
