"""缓存服务

使用内存缓存提高性能，支持 TTL 过期和装饰器模式。
"""
import time
import hashlib
from typing import Optional, Any, Callable
from functools import wraps

from config.logging import get_logger


logger = get_logger(__name__)


class SimpleCache:
    """简单的内存缓存实现"""

    def __init__(self, default_ttl: int = 300):
        """
        Args:
            default_ttl: 默认缓存过期时间（秒）
        """
        self._cache: dict[str, tuple[Any, float]] = {}  # {key: (value, expire_time)}
        self._default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if key not in self._cache:
            return None

        value, expire_time = self._cache[key]
        if time.time() > expire_time:
            # 已过期，删除
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存值"""
        if ttl is None:
            ttl = self._default_ttl
        expire_time = time.time() + ttl
        self._cache[key] = (value, expire_time)

    def delete(self, key: str) -> None:
        """删除缓存"""
        if key in self._cache:
            del self._cache[key]

    def clear(self) -> None:
        """清空所有缓存"""
        self._cache.clear()

    def cleanup(self) -> None:
        """清理过期的缓存"""
        now = time.time()
        expired_keys = [
            key for key, (_, expire_time) in self._cache.items()
            if now > expire_time
        ]
        for key in expired_keys:
            del self._cache[key]


# 全局缓存实例
_cache = SimpleCache()


def get_cache() -> SimpleCache:
    """获取缓存实例"""
    return _cache


def cached(ttl: int = 300, key_prefix: str = ""):
    """
    缓存装饰器

    Args:
        ttl: 缓存过期时间（秒）
        key_prefix: 缓存键前缀
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = _generate_cache_key(key_prefix, func.__name__, args, kwargs)

            # 尝试从缓存获取
            cached_value = _cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"[CACHE] Hit: {cache_key}")
                return cached_value

            # 执行函数
            result = await func(*args, **kwargs)

            # 存入缓存
            _cache.set(cache_key, result, ttl)
            logger.debug(f"[CACHE] Set: {cache_key}")

            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = _generate_cache_key(key_prefix, func.__name__, args, kwargs)

            # 尝试从缓存获取
            cached_value = _cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"[CACHE] Hit: {cache_key}")
                return cached_value

            # 执行函数
            result = func(*args, **kwargs)

            # 存入缓存
            _cache.set(cache_key, result, ttl)
            logger.debug(f"[CACHE] Set: {cache_key}")

            return result

        # 根据函数类型返回对应的包装器
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def _generate_cache_key(prefix: str, func_name: str, args: tuple, kwargs: dict) -> str:
    """生成缓存键"""
    # 将参数序列化为字符串
    key_parts = [prefix, func_name]

    # 处理位置参数（跳过 self 参数）
    if args and hasattr(args[0], '__class__'):
        # 如果第一个参数是 self，跳过它
        args = args[1:]

    if args:
        key_parts.append(str(args))

    if kwargs:
        # 对 kwargs 排序以确保一致性
        sorted_kwargs = sorted(kwargs.items())
        key_parts.append(str(sorted_kwargs))

    # 生成哈希
    key_str = ":".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


def invalidate_pattern(pattern: str) -> None:
    """根据模式前缀删除缓存"""
    keys_to_delete = [
        key for key in _cache._cache.keys()
        if key.startswith(pattern)
    ]
    for key in keys_to_delete:
        _cache.delete(key)
    logger.info(f"[CACHE] Invalidated {len(keys_to_delete)} entries: {pattern}")
