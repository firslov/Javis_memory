"""记忆管理 API 路由

提供 RAG 向量检索记忆系统的搜索、同步和状态查询接口。
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, FileResponse
from config.settings import get_settings
from services.memory_index import MemoryIndexManager
from api.routes.dependencies import get_or_create_memory_manager
from pathlib import Path

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

@router.get("/viz", response_class=HTMLResponse)
async def memory_visualization_page():
    """记忆系统可视化页面"""
    template_path = Path(__file__).parent.parent.parent / "templates" / "memory_viz.html"
    if template_path.exists():
        return FileResponse(template_path)
    return HTMLResponse("<h1>模板文件不存在</h1>")

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


# ============================================================================
# Memory Lifecycle Management API
# ============================================================================

@router.get("/stats")
async def get_memory_stats(
    user_id: int = Query(1, description="User ID"),
    http_request: Request = Request,
):
    """获取记忆统计信息

    返回各层级记忆数量、平均重要性分数、总访问次数等统计数据。
    """
    settings = get_settings()

    if not settings.memory_search.enabled:
        raise HTTPException(status_code=503, detail="Memory search is disabled")

    if not settings.memory_search.lifecycle.enabled:
        raise HTTPException(status_code=400, detail="Memory lifecycle management is disabled")

    manager: MemoryIndexManager = await get_or_create_memory_manager(http_request, user_id)

    try:
        if hasattr(manager, '_lifecycle_manager') and manager._lifecycle_manager:
            stats = await manager._lifecycle_manager.get_memory_stats()
            return stats
        else:
            # 如果生命周期管理器未初始化，返回基本统计
            cursor = await manager._db.execute("""
                SELECT COUNT(*) FROM memory_chunks WHERE user_id = ?
            """, (user_id,))
            row = await cursor.fetchone()
            await cursor.close()
            return {
                "user_id": user_id,
                "tiers": {},
                "total_chunks": row[0] if row else 0,
                "avg_importance": 0.0,
                "total_accesses": 0,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats failed: {str(e)}")


@router.post("/consolidate")
async def consolidate_memories(
    user_id: int = Query(1, description="User ID"),
    http_request: Request = Request,
):
    """手动触发记忆合并

    检测相似记忆簇并合并为更高层级的摘要。
    """
    settings = get_settings()

    if not settings.memory_search.enabled:
        raise HTTPException(status_code=503, detail="Memory search is disabled")

    if not settings.memory_search.consolidation.enabled:
        raise HTTPException(status_code=400, detail="Memory consolidation is disabled")

    manager: MemoryIndexManager = await get_or_create_memory_manager(http_request, user_id)

    try:
        from services.memory_consolidation import MemoryConsolidator

        consolidator = MemoryConsolidator(
            db_conn=manager._db,
            user_id=user_id,
            config=settings.memory_search.consolidation,
        )

        result = await consolidator.run_consolidation()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Consolidation failed: {str(e)}")


@router.post("/cleanup")
async def cleanup_working_memory(
    user_id: int = Query(1, description="User ID"),
    http_request: Request = Request,
):
    """清理过期的工作记忆

    将超过配置天数的工作记忆移动到档案层级。
    """
    settings = get_settings()

    if not settings.memory_search.enabled:
        raise HTTPException(status_code=503, detail="Memory search is disabled")

    if not settings.memory_search.lifecycle.enabled:
        raise HTTPException(status_code=400, detail="Memory lifecycle management is disabled")

    manager: MemoryIndexManager = await get_or_create_memory_manager(http_request, user_id)

    try:
        if hasattr(manager, '_lifecycle_manager') and manager._lifecycle_manager:
            result = await manager._lifecycle_manager.cleanup_working_memory()
            return result
        else:
            return {"cleaned": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.post("/promote")
async def promote_memories(
    user_id: int = Query(1, description="User ID"),
    http_request: Request = Request,
):
    """提升记忆层级

    将高访问次数的工作记忆提升为长期记忆。
    """
    settings = get_settings()

    if not settings.memory_search.enabled:
        raise HTTPException(status_code=503, detail="Memory search is disabled")

    if not settings.memory_search.lifecycle.enabled:
        raise HTTPException(status_code=400, detail="Memory lifecycle management is disabled")

    manager: MemoryIndexManager = await get_or_create_memory_manager(http_request, user_id)

    try:
        if hasattr(manager, '_lifecycle_manager') and manager._lifecycle_manager:
            result = await manager._lifecycle_manager.promote_memories()
            return result
        else:
            return {"promoted": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Promotion failed: {str(e)}")


# ============================================================================
# Memory Visualization API
# ============================================================================

@router.get("/visualization")
async def get_memory_visualization(
    user_id: int = Query(1, description="User ID"),
    limit: int = Query(10, description="Limit for top memories", ge=1, le=50),
    http_request: Request = Request,
):
    """获取记忆可视化数据

    返回用于可视化展示的完整数据，包括：
    - 层级分布
    - 重要性分布
    - Top重要记忆
    - 最近访问记录
    """
    settings = get_settings()

    if not settings.memory_search.enabled:
        raise HTTPException(status_code=503, detail="Memory search is disabled")

    manager: MemoryIndexManager = await get_or_create_memory_manager(http_request, user_id)

    try:
        from services.memory_visualization import MemoryVisualizer

        visualizer = MemoryVisualizer(manager._db)
        data = await visualizer.collect_data(user_id)

        return {
            "user_id": data.user_id,
            "overview": {
                "total_chunks": data.total_chunks,
                "avg_importance": data.avg_importance,
                "total_accesses": data.total_accesses,
            },
            "tier_distribution": data.tier_stats,
            "importance_distribution": data.importance_distribution,
            "top_memories": data.top_memories[:limit],
            "recent_accesses": data.recent_accesses[:limit],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")
