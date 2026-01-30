"""对话管理服务

提供对话和消息的 CRUD 操作。
"""
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import User, Conversation, Message
from database.repository import (
    UserRepository,
    ConversationRepository,
    MessageRepository,
)
from config.settings import get_settings
from config.logging import get_logger
from services.cache import get_cache


logger = get_logger(__name__)


class ConversationService:
    """对话管理服务"""

    def __init__(self, session: AsyncSession):
        self.session = session
        self._settings = get_settings()
        self._cache = get_cache()

    # ==================== 用户管理 ====================

    async def get_or_create_user(
        self,
        name: str,
        email: str = None
    ) -> User:
        """获取或创建用户"""
        return await UserRepository.get_or_create(self.session, name, email)

    # ==================== 对话管理 ====================

    async def create_conversation(
        self,
        user_id: int,
        model: str,
        title: str = "新对话"
    ) -> Conversation:
        """创建对话"""
        return await ConversationRepository.create(
            self.session,
            user_id,
            model,
            title
        )

    async def get_conversation(
        self,
        conversation_id: int
    ) -> Conversation:
        """获取对话"""
        return await ConversationRepository.get_by_id(self.session, conversation_id)

    async def get_recent_conversations(
        self,
        user_id: int,
        limit: int = 10
    ) -> List[Conversation]:
        """获取最近的对话"""
        return await ConversationRepository.get_recent_conversations(
            self.session,
            user_id,
            limit
        )

    # ==================== 消息管理 ====================

    async def save_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        tokens: int = None
    ) -> Message:
        """保存消息"""
        return await MessageRepository.create(
            self.session,
            conversation_id,
            role,
            content,
            tokens
        )

    async def get_conversation_messages(
        self,
        conversation_id: int
    ) -> List[Message]:
        """获取对话的所有消息"""
        return await MessageRepository.get_by_conversation(
            self.session,
            conversation_id
        )

    async def get_conversation_messages_dict(
        self,
        conversation_id: int
    ) -> List[dict]:
        """获取对话的消息列表（字典格式，用于LLM）"""
        messages = await self.get_conversation_messages(conversation_id)
        result = []
        for msg in messages:
            content = msg.content
            # 过滤掉只有emoji/标点符号的"假"消息
            has_real_content = any(c.isalnum() or c.strip() for c in content)
            if has_real_content:
                result.append({
                    "role": msg.role,
                    "content": msg.content
                })
        return result
