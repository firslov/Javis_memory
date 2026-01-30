"""数据访问层

提供用户、对话、消息、API Key 的数据访问操作。
"""
from typing import Optional, List
from datetime import datetime
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from .models import User, Conversation, Message, APIKey


# ============================================================================
# 基础数据访问
# ============================================================================

class UserRepository:
    """用户数据访问"""

    @staticmethod
    async def get_or_create(
        session: AsyncSession,
        name: str,
        email: Optional[str] = None
    ) -> User:
        """获取或创建用户"""
        result = await session.execute(
            select(User).where(User.name == name)
        )
        user = result.scalar_one_or_none()

        if not user:
            user = User(name=name, email=email)
            session.add(user)
            await session.flush()

        return user

    @staticmethod
    async def get_by_id(session: AsyncSession, user_id: int) -> Optional[User]:
        """根据ID获取用户"""
        return await session.get(User, user_id)


class ConversationRepository:
    """对话数据访问"""

    @staticmethod
    async def create(
        session: AsyncSession,
        user_id: int,
        model: str,
        title: str = "新对话"
    ) -> Conversation:
        """创建对话"""
        conversation = Conversation(
            user_id=user_id,
            model=model,
            title=title
        )
        session.add(conversation)
        await session.flush()
        return conversation

    @staticmethod
    async def get_by_id(session: AsyncSession, conversation_id: int) -> Optional[Conversation]:
        """根据ID获取对话"""
        return await session.get(Conversation, conversation_id)

    @staticmethod
    async def get_recent_conversations(
        session: AsyncSession,
        user_id: int,
        limit: int = 10
    ) -> List[Conversation]:
        """获取最近的对话"""
        result = await session.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(desc(Conversation.updated_at))
            .limit(limit)
        )
        return list(result.scalars().all())


class MessageRepository:
    """消息数据访问"""

    @staticmethod
    async def create(
        session: AsyncSession,
        conversation_id: int,
        role: str,
        content: str,
        tokens: Optional[int] = None
    ) -> Message:
        """创建消息"""
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tokens=tokens
        )
        session.add(message)
        await session.flush()

        # 更新对话的更新时间
        await session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        return message

    @staticmethod
    async def get_by_conversation(
        session: AsyncSession,
        conversation_id: int
    ) -> List[Message]:
        """获取对话的所有消息"""
        result = await session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )
        return list(result.scalars().all())


# ============================================================================
# API Key 管理数据访问
# ============================================================================

class APIKeyRepository:
    """API密钥数据访问"""

    @staticmethod
    async def create(
        session: AsyncSession,
        user_id: int,
        name: str,
        key: str,
        rate_limit: Optional[int] = None,
        daily_limit: Optional[int] = None,
        expires_at: Optional[datetime] = None
    ) -> APIKey:
        """创建API密钥"""
        api_key = APIKey(
            user_id=user_id,
            name=name,
            key=key,
            rate_limit=rate_limit,
            daily_limit=daily_limit,
            expires_at=expires_at
        )
        session.add(api_key)
        await session.flush()
        return api_key

    @staticmethod
    async def get_by_key(session: AsyncSession, key: str) -> Optional[APIKey]:
        """根据key获取API密钥"""
        result = await session.execute(
            select(APIKey)
            .where(APIKey.key == key)
            .where(APIKey.is_active == True)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_user(session: AsyncSession, user_id: int) -> List[APIKey]:
        """获取用户的所有API密钥"""
        result = await session.execute(
            select(APIKey)
            .where(APIKey.user_id == user_id)
            .order_by(desc(APIKey.created_at))
        )
        return list(result.scalars().all())

    @staticmethod
    async def update_last_used(session: AsyncSession, api_key: APIKey) -> None:
        """更新API密钥最后使用时间"""
        api_key.last_used_at = datetime.utcnow()
        api_key.total_requests += 1
        await session.flush()

    @staticmethod
    def is_valid(api_key: APIKey) -> bool:
        """检查API密钥是否有效（激活且未过期）"""
        if not api_key.is_active:
            return False
        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            return False
        return True

    @staticmethod
    async def update_usage(session: AsyncSession, api_key: APIKey) -> None:
        """更新API密钥使用统计"""
        api_key.last_used_at = datetime.utcnow()
        api_key.total_requests += 1
        await session.flush()

    @staticmethod
    async def check_rate_limit(session: AsyncSession, api_key: APIKey) -> tuple[bool, str]:
        """检查速率限制"""
        if api_key.daily_limit and api_key.total_requests >= api_key.daily_limit:
            return False, f"Daily limit of {api_key.daily_limit} requests exceeded"
        return True, ""
