"""聊天 API 路由

提供聊天完成和模型列表接口，集成 RAG 向量检索记忆系统。
对话后会自动保存摘要到记忆文件。
"""
import json
import asyncio
import time
from typing import Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Security, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from database.session import get_db_session
from database.models import User
from config.settings import get_settings
from config.logging import get_logger
from services.conversation import ConversationService
from services.memory_index import MemoryIndexManager
from services.search_engine import format_search_results
from services.auth import AuthService
from services.llm_forwarder import LLMForwarder
from schemas.chat import ChatCompletionRequest, ModelsResponse
from api.routes.dependencies import get_llm_forwarder, get_or_create_memory_manager


logger = get_logger(__name__)


router = APIRouter(prefix="/v1", tags=["chat"])
security = HTTPBearer(auto_error=False)


async def optional_auth(
    credentials: HTTPAuthorizationCredentials = Security(security),
    session: AsyncSession = Depends(get_db_session)
) -> Optional[User]:
    """可选认证依赖

    数据库错误时优雅降级为匿名用户，而不是让服务崩溃。
    """
    if not credentials:
        return None
    try:
        api_key_str = credentials.credentials
        _, user = await AuthService.validate_api_key(session, api_key_str)
        return user
    except HTTPException:
        # API Key 无效（格式错误、已过期等）
        return None
    except SQLAlchemyError as e:
        # 数据库错误（表不存在、连接失败等）
        # 记录日志但返回匿名用户，避免服务崩溃
        logger.warning(f"[AUTH] Database error during authentication: {e}")
        return None


async def build_context_with_rag_memory(
    user_id: int,
    user_message: str,
    messages: list,
    settings,
    request: Request,
) -> list:
    """使用 RAG 向量检索构建增强上下文"""
    if not settings.memory_search.enabled:
        return [msg.model_dump() for msg in messages]

    try:
        manager: MemoryIndexManager = await get_or_create_memory_manager(request, user_id)

        # Sync if configured
        if settings.memory_search.sync.on_session_start:
            await manager.sync()

        # Search for relevant memories
        search_results = await manager.search(
            query=user_message,
            max_results=settings.memory_search.query.max_results,
            min_score=settings.memory_search.query.min_score,
        )

        if search_results:
            memory_context = format_search_results(search_results)
            logger.debug(f"[RAG] Found {len(search_results)} memories")

            # Build enhanced messages with memory context
            enhanced_messages = []

            # Add system message with memory context
            system_message = {
                "role": "system",
                "content": (
                    "You are an AI assistant with access to relevant memories from past conversations. "
                    "Use these memories to provide more personalized and contextual responses.\n\n"
                    f"{memory_context}"
                )
            }
            enhanced_messages.append(system_message)

            # Add original messages
            for msg in messages:
                enhanced_messages.append(msg.model_dump())

            return enhanced_messages
        else:
            logger.debug("[RAG] No relevant memories")
            return [msg.model_dump() for msg in messages]

    except Exception as e:
        logger.error(f"[RAG] Search failed: {e}")
        return [msg.model_dump() for msg in messages]


async def save_conversation_to_memory(
    user_id: int,
    conversation_id: int,
    session: AsyncSession,
    request: Request,
):
    """保存对话摘要到 RAG 记忆文件"""
    from pathlib import Path

    conversation_service = ConversationService(session)
    messages = await conversation_service.get_conversation_messages(conversation_id)

    if not messages:
        return

    # 生成对话摘要
    summary_lines = [
        f"# 对话记录 - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    # 添加对话内容
    for msg in messages:
        if msg.role == "user":
            summary_lines.append(f"**用户**: {msg.content}")
        elif msg.role == "assistant":
            summary_lines.append(f"**助手**: {msg.content[:200]}..." if len(msg.content) > 200 else f"**助手**: {msg.content}")
        summary_lines.append("")

    # 保存到用户专用记忆目录
    settings = get_settings()
    memory_dir = Path(settings.memory_search.memory_files_dir.format(user_id=user_id)).expanduser()
    memory_dir.mkdir(parents=True, exist_ok=True)

    # 使用用户专用的记忆文件
    memory_file = memory_dir / "conversations.md"

    # 追加到文件
    with open(memory_file, "a", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
        f.write("\n---\n\n")

    logger.debug(f"[RAG] Saved summary to {memory_file}")

    # 触发同步
    try:
        manager: MemoryIndexManager = await get_or_create_memory_manager(request, user_id)
        await manager.sync(force=True)
    except Exception as e:
        logger.error(f"[RAG] Sync failed: {e}")


@router.post("/chat/completions")
async def chat_completions(
    req_data: ChatCompletionRequest,
    current_user: Optional[User] = Depends(optional_auth),
    session: AsyncSession = Depends(get_db_session),
    llm_forwarder: LLMForwarder = Depends(get_llm_forwarder),
    request: Request = Request,
):
    """聊天完成接口 - 支持 RAG 向量检索记忆系统"""
    settings = get_settings()
    conversation_service = ConversationService(session)

    # 1. 获取用户
    if current_user:
        user_id = current_user.id
    else:
        user = await conversation_service.get_or_create_user(
            name=f"user_{req_data.user_id or 'default'}",
        )
        user_id = user.id

    # 2. 获取或创建对话
    conversation_id = req_data.conversation_id
    if not conversation_id:
        # 尝试获取用户的最近对话
        recent_conversations = await conversation_service.get_recent_conversations(user_id, limit=1)
        if recent_conversations:
            # 检查最近对话是否在1小时内
            recent_conv = recent_conversations[0]
            if recent_conv.updated_at and (datetime.utcnow() - recent_conv.updated_at) < timedelta(hours=1):
                conversation_id = recent_conv.id
            else:
                # 超过1小时，创建新对话
                conversation = await conversation_service.create_conversation(
                    user_id=user_id,
                    model=req_data.model
                )
                conversation_id = conversation.id
        else:
            # 没有历史对话，创建新对话
            conversation = await conversation_service.create_conversation(
                user_id=user_id,
                model=req_data.model
            )
            conversation_id = conversation.id

    # 3. 提取用户消息
    user_message = req_data.messages[-1].content if req_data.messages else ""

    # 4. 构建增强上下文（使用 RAG 系统）
    if req_data.enable_profile:
        enhanced_messages = await build_context_with_rag_memory(
            user_id=user_id,
            user_message=user_message,
            messages=req_data.messages,
            settings=settings,
            request=request,
        )
    else:
        enhanced_messages = [msg.model_dump() for msg in req_data.messages]

    # 5. 保存用户消息
    await conversation_service.save_message(
        conversation_id=conversation_id,
        role="user",
        content=user_message
    )
    await session.commit()

    # 6. 准备LLM请求
    llm_request = {
        "model": req_data.model,
        "messages": enhanced_messages,
        "temperature": req_data.temperature,
    }
    if req_data.max_tokens:
        llm_request["max_tokens"] = req_data.max_tokens
    if req_data.stream:
        llm_request["stream"] = True

    # 7. 发送请求
    if req_data.stream:
        async def stream_wrapper():
            full_response = ""
            async for chunk in llm_forwarder.forward_stream(
                model=req_data.model,
                data=llm_request
            ):
                yield chunk
                # Parse SSE events from chunk
                for line in chunk.split('\n'):
                    line = line.strip()
                    if line.startswith('data: '):
                        data_str = line[6:].strip()
                        if data_str == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and data['choices']:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    full_response += content
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.debug(f"[LLM] SSE parse error: {e}")

            # 异步处理记忆更新
            asyncio.create_task(_save_assistant_response_and_memory(
                conversation_id, user_id, full_response, req_data.model, request
            ))

        return StreamingResponse(
            stream_wrapper(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        response_text = await llm_forwarder.forward_request(
            model=req_data.model,
            data=llm_request
        )

        response = json.loads(response_text)

        assistant_message = ""
        if "choices" in response and response["choices"]:
            assistant_message = response["choices"][0].get("message", {}).get("content", "")

        await conversation_service.save_message(
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_message
        )

        # 处理记忆更新
        await save_conversation_to_memory(user_id, conversation_id, session, request)

        await session.commit()
        return response


async def _save_assistant_response_and_memory(
    conversation_id: int,
    user_id: int,
    content: str,
    model: str,
    request: Request,
):
    """异步保存助手回复并处理记忆更新"""
    async for session in get_db_session():
        try:
            conversation_service = ConversationService(session)

            await conversation_service.save_message(
                conversation_id=conversation_id,
                role="assistant",
                content=content
            )

            # 处理记忆更新
            await save_conversation_to_memory(user_id, conversation_id, session, request)

            await session.commit()
        except Exception as e:
            await session.rollback()
            logging.getLogger(__name__).error(f"[RAG] Save error: {e}")


@router.get("/models")
async def list_models(
    llm_forwarder: LLMForwarder = Depends(get_llm_forwarder)
) -> ModelsResponse:
    """获取可用模型列表"""
    models = llm_forwarder.get_available_models()

    return ModelsResponse(
        data=[
            {
                "id": model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "javis"
            }
            for model in models
        ]
    )
