"""OpenAI 兼容 API 嵌入客户端

支持 OpenAI、DeepSeek、豆包等兼容 OpenAI API 格式的服务。
"""
import asyncio
from typing import List

import httpx

from .base import EmbeddingProvider
from config.logging import get_logger


logger = get_logger(__name__)


# Default vector dimensions for common models
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # DeepSeek (uses OpenAI-compatible API)
    "deepseek-embed": 1536,
    # Doubao (ByteDance embedding model)
    "doubao-embedding": 2560,
}


class OpenAIEmbeddingClient(EmbeddingProvider):
    """OpenAI-compatible API embedding client.

    Supports:
    - OpenAI API
    - DeepSeek API
    - Any OpenAI-compatible API
    """

    # Default model dimensions
    DEFAULT_DIMENSION = 1536

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        timeout: float = 60.0,
    ):
        """Initialize the OpenAI embedding client.

        Args:
            api_key: API key for authentication.
            base_url: Base URL of the API (default: OpenAI).
            model: Model name to use for embeddings.
            timeout: Request timeout in seconds.
        """
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

        # Determine vector dimension
        self._dimension = MODEL_DIMENSIONS.get(model, self.DEFAULT_DIMENSION)

        # HTTP client with connection pooling
        self._client: httpx.AsyncClient | None = None

    @classmethod
    async def create(cls, config) -> "OpenAIEmbeddingClient":
        """Create an OpenAI embedding client from config.

        Args:
            config: Memory search configuration.

        Returns:
            Initialized OpenAIEmbeddingClient instance.
        """
        from config.settings import get_settings

        settings = get_settings()

        # Determine API key and base URL
        api_key = None
        base_url = "https://api.openai.com/v1"
        model = config.model or "text-embedding-3-small"

        # Check remote config
        if config.remote:
            if config.remote.api_key:
                api_key = config.remote.api_key
            if config.remote.base_url:
                base_url = config.remote.base_url

        # Fall back to server config
        if not api_key:
            # Try to find a server with the configured model
            for server_name, server_config in settings.servers.items():
                if model in server_config.models:
                    api_key = server_config.api_key
                    base_url = server_config.base_url
                    break

        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set:\n"
                "  - memory_search.remote.api_key in config/servers.yaml"
            )

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self._timeout),
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=10,
                    keepalive_expiry=30.0,
                ),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in batch.

        Args:
            texts: List of texts to embed.

        Returns:
            A list of embedding vectors.
        """
        if not texts:
            return []

        client = await self._get_client()

        # OpenAI supports batch embedding
        # Maximum batch size is typically 2048
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            try:
                response = await client.post(
                    "/embeddings",
                    json={
                        "input": batch,
                        "model": self._model,
                    },
                )
                response.raise_for_status()

                data = response.json()
                batch_embeddings = [item["embedding"] for item in data["data"]]
                all_embeddings.extend(batch_embeddings)

            except httpx.HTTPStatusError as e:
                logger.error(f"[EMBED] Request failed: {e}")
                raise
            except Exception as e:
                logger.error(f"[EMBED] Error: {e}")
                raise

        return all_embeddings

    @property
    def id(self) -> str:
        """Unique identifier for this provider instance."""
        return f"openai:{self._model}"

    @property
    def model(self) -> str:
        """Model name used for embeddings."""
        return self._model

    @property
    def vector_dimension(self) -> int:
        """Dimension of the output embedding vectors."""
        return self._dimension
