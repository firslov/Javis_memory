"""API Key 管理路由

提供 API Key 的创建、查询、停用和删除功能。
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_db_session
from database.models import User, APIKey
from database.repository import APIKeyRepository
from services.auth import AuthService
from schemas.chat import APIKeyCreate, APIKeyResponse, APIKeyInfo


router = APIRouter(prefix="/v1/api-keys", tags=["api-keys"])
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: AsyncSession = Depends(get_db_session)
) -> User:
    """获取当前认证用户"""
    api_key_str = credentials.credentials
    api_key, user = await AuthService.validate_api_key(session, api_key_str)
    return user


@router.post("", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    data: APIKeyCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session)
):
    """创建新的API密钥"""
    api_key = await AuthService.create_api_key(
        session,
        user_id=current_user.id,
        name=data.name,
        rate_limit=data.rate_limit,
        daily_limit=data.daily_limit,
        expires_days=data.expires_days
    )
    await session.commit()

    return APIKeyResponse(
        id=api_key.id,
        name=api_key.name,
        key=api_key.key,  # 只在创建时返回完整密钥
        is_active=api_key.is_active,
        rate_limit=api_key.rate_limit,
        daily_limit=api_key.daily_limit,
        expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None,
        last_used_at=api_key.last_used_at.isoformat() if api_key.last_used_at else None,
        total_requests=api_key.total_requests,
        created_at=api_key.created_at.isoformat(),
    )


@router.get("", response_model=list[APIKeyInfo])
async def list_api_keys(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session)
):
    """获取当前用户的所有API密钥"""
    api_keys = await APIKeyRepository.get_user_keys(session, current_user.id)

    return [
        APIKeyInfo(
            id=key.id,
            name=key.name,
            key_preview=f"{key.key[:8]}...{key.key[-4:]}" if key.key else "",
            is_active=key.is_active,
            rate_limit=key.rate_limit,
            daily_limit=key.daily_limit,
            expires_at=key.expires_at.isoformat() if key.expires_at else None,
            last_used_at=key.last_used_at.isoformat() if key.last_used_at else None,
            total_requests=key.total_requests,
            created_at=key.created_at.isoformat(),
        )
        for key in api_keys
    ]


@router.post("/{key_id}/deactivate", response_model=APIKeyInfo)
async def deactivate_api_key(
    key_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session)
):
    """停用API密钥"""
    api_key = await session.get(APIKey, key_id)
    if not api_key or api_key.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )

    updated_key = await APIKeyRepository.deactivate(session, key_id)
    await session.commit()

    return APIKeyInfo(
        id=updated_key.id,
        name=updated_key.name,
        key_preview=f"{updated_key.key[:8]}...{updated_key.key[-4:]}",
        is_active=updated_key.is_active,
        rate_limit=updated_key.rate_limit,
        daily_limit=updated_key.daily_limit,
        expires_at=updated_key.expires_at.isoformat() if updated_key.expires_at else None,
        last_used_at=updated_key.last_used_at.isoformat() if updated_key.last_used_at else None,
        total_requests=updated_key.total_requests,
        created_at=updated_key.created_at.isoformat(),
    )


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    key_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session)
):
    """删除API密钥"""
    api_key = await session.get(APIKey, key_id)
    if not api_key or api_key.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )

    await APIKeyRepository.delete(session, key_id)
    await session.commit()

    return None
