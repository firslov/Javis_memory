"""嵌入服务提供者工厂

根据配置创建相应的嵌入服务提供者实例。
"""
from typing import Optional, TYPE_CHECKING

from .base import EmbeddingProvider
from config.logging import get_logger

if TYPE_CHECKING:
    from config.settings import MemorySearchConfig

logger = get_logger(__name__)


async def create_embedding_provider(config: "MemorySearchConfig") -> EmbeddingProvider:
    """Create an embedding provider based on configuration.

    Provider selection logic for "auto":
    1. Check if local model file exists → use local
    2. Try OpenAI API
    3. Try Gemini API
    4. Try fallback provider
    5. Raise error if all fail

    Args:
        config: Memory search configuration.

    Returns:
        An initialized EmbeddingProvider instance.

    Raises:
        ValueError: If no suitable provider is available.
    """
    from .openai_client import OpenAIEmbeddingClient
    from .gemini_client import GeminiEmbeddingClient
    from .local_client import LocalEmbeddingClient

    provider = config.provider.lower()

    # Auto-detect provider
    if provider == "auto":
        provider = await _detect_provider(config)
        logger.info(f"[EMBED] Auto-detected provider: {provider}")

    # Create provider instance
    if provider == "openai":
        return await OpenAIEmbeddingClient.create(config)
    elif provider == "gemini":
        return await GeminiEmbeddingClient.create(config)
    elif provider == "local":
        return await LocalEmbeddingClient.create(config)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


async def _detect_provider(config: "MemorySearchConfig") -> str:
    """Detect the best available embedding provider.

    Returns:
        Provider name: "local", "openai", "gemini", or error.
    """
    import os

    from .local_client import LocalEmbeddingClient

    # 1. Check for local model
    if config.local.model_path:
        model_path = os.path.expanduser(config.local.model_path)
        if os.path.exists(model_path):
            logger.info("[EMBED] Found local model")
            return "local"

    # Check default local model path
    default_local_path = os.path.expanduser("~/.cache/javis/embeddings")
    if os.path.exists(default_local_path):
        logger.info("[EMBED] Found default local models")
        return "local"

    # 2. Try OpenAI API
    if config.remote and config.remote.api_key:
        try:
            client = await OpenAIEmbeddingClient.create(config)
            # Test connection
            await client.embed_query("test")
            await client.close()
            logger.info("[EMBED] OpenAI API available")
            return "openai"
        except Exception as e:
            logger.debug(f"[EMBED] OpenAI unavailable: {e}")

    # 3. Try Gemini API
    if config.remote and config.remote.gemini_api_key:
        try:
            from .gemini_client import GeminiEmbeddingClient
            client = await GeminiEmbeddingClient.create(config)
            await client.embed_query("test")
            await client.close()
            logger.info("[EMBED] Gemini API available")
            return "gemini"
        except Exception as e:
            logger.debug(f"[EMBED] Gemini unavailable: {e}")

    # 4. Try fallback
    fallback = config.fallback.lower() if config.fallback else "none"
    if fallback != "none":
        logger.info(f"[EMBED] Using fallback: {fallback}")
        return fallback

    # 5. Check if sentence-transformers is available (local fallback)
    try:
        import sentence_transformers
        logger.info("[EMBED] sentence-transformers available")
        return "local"
    except ImportError:
        pass

    raise ValueError(
        "No embedding provider available. Please configure one of:\n"
        "  - OpenAI API key in config/servers.yaml\n"
        "  - Gemini API key in config/servers.yaml\n"
        "  - Local model path in config/servers.yaml\n"
        "  - Install sentence-transformers: pip install sentence-transformers"
    )


# Import at bottom to avoid circular dependency
from .openai_client import OpenAIEmbeddingClient
