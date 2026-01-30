"""认证服务

提供 API Key 验证、生成和管理功能。
"""
import secrets
from typing import Optional, Tuple
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import APIKey, User
from database.repository import APIKeyRepository, UserRepository
from config.logging import get_logger


logger = get_logger(__name__)

# HTTP Bearer 认证方案
security = HTTPBearer(auto_error=False)


class AuthService:
    """认证服务"""

    @staticmethod
    def generate_api_key() -> str:
        """生成随机API密钥 (sk-开头，48字符)"""
        return f"sk-{secrets.token_urlsafe(36)}"

    @staticmethod
    async def create_api_key(
        session: AsyncSession,
        user_id: int,
        name: str,
        rate_limit: Optional[int] = None,
        daily_limit: Optional[int] = None,
        expires_days: Optional[int] = None
    ) -> APIKey:
        """创建API密钥"""
        from datetime import datetime, timedelta

        key = AuthService.generate_api_key()
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)

        api_key = await APIKeyRepository.create(
            session,
            user_id=user_id,
            key=key,
            name=name,
            rate_limit=rate_limit,
            daily_limit=daily_limit,
            expires_at=expires_at
        )

        logger.info(f"[AUTH] API key '{name}' created for user {user_id}")
        return api_key

    @staticmethod
    async def validate_api_key(
        session: AsyncSession,
        api_key_str: str
    ) -> Tuple[APIKey, User]:
        """
        验证API密钥

        Returns:
            (APIKey, User)

        Raises:
            HTTPException: 认证失败时
        """
        if not api_key_str:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key is required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # 获取API密钥记录
        api_key = await APIKeyRepository.get_by_key(session, api_key_str)
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # 检查密钥是否有效
        if not APIKeyRepository.is_valid(api_key):
            if not api_key.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="API key is deactivated",
                )
            # Must be expired
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key has expired",
            )

        # 检查速率限制
        passed, error_msg = await APIKeyRepository.check_rate_limit(session, api_key)
        if not passed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=error_msg,
            )

        # 获取用户信息
        user = await UserRepository.get_by_id(session, api_key.user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )

        # 更新使用统计
        await APIKeyRepository.update_usage(session, api_key)

        return api_key, user
