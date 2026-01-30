"""Gemini API 嵌入客户端

使用 Google Gemini API 生成文本嵌入向量。
"""
from typing import List

import httpx

from .base import EmbeddingProvider
from config.logging import get_logger


logger = get_logger(__name__)


# Gemini embedding models
GEMINI_MODELS = {
    "text-embedding-004": "models/text-embedding-004",
    "embedding-001": "models/embedding-001",
}

# Vector dimensions for Gemini models
GEMINI_DIMENSIONS = {
    "text-embedding-004": 768,
    "embedding-001": 768,
}


class GeminiEmbeddingClient(EmbeddingProvider):
    """Google Gemini API embedding client."""

    DEFAULT_MODEL = "text-embedding-004"
    DEFAULT_DIMENSION = 768
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-004",
        timeout: float = 60.0,
    ):
        """Initialize the Gemini embedding client.

        Args:
            api_key: Gemini API key.
            model: Model name to use for embeddings.
            timeout: Request timeout in seconds.
        """
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._dimension = GEMINI_DIMENSIONS.get(model, self.DEFAULT_DIMENSION)

        # HTTP client
        self._client: httpx.AsyncClient | None = None

    @classmethod
    async def create(cls, config) -> "GeminiEmbeddingClient":
        """Create a Gemini embedding client from config.

        Args:
            config: Memory search configuration.

        Returns:
            Initialized GeminiEmbeddingClient instance.
        """
        api_key = None
        model = config.model or cls.DEFAULT_MODEL

        # Check remote config for Gemini API key
        if config.remote and config.remote.gemini_api_key:
            api_key = config.remote.gemini_api_key

        if not api_key:
            raise ValueError(
                "Gemini API key not found. Please set:\n"
                "  - memory_search.remote.gemini_api_key in config/servers.yaml"
            )

        return cls(
            api_key=api_key,
            model=model,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self._timeout),
                params={"key": self._api_key},
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

        Note: Gemini API doesn't support batch requests,
        so we make individual requests concurrently.

        Args:
            texts: List of texts to embed.

        Returns:
            A list of embedding vectors.
        """
        if not texts:
            return []

        import asyncio

        client = await self._get_client()
        model_path = GEMINI_MODELS.get(self._model, f"models/{self._model}")

        # Make concurrent requests
        tasks = [self._embed_one(client, model_path, text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle errors
        embeddings = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"[EMBED] Failed to embed text {i}: {result}")
                # Return zero vector on error
                embeddings.append([0.0] * self._dimension)
            else:
                embeddings.append(result)

        return embeddings

    async def _embed_one(
        self, client: httpx.AsyncClient, model_path: str, text: str
    ) -> List[float]:
        """Embed a single text.

        Args:
            client: HTTP client.
            model_path: Model API path.
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        try:
            response = await client.post(
                f"{model_path}:embedContent",
                json={
                    "content": {
                        "parts": [{"text": text}]
                    }
                },
            )
            response.raise_for_status()

            data = response.json()
            embedding = data["embedding"]["values"]
            return embedding

        except httpx.HTTPStatusError as e:
            logger.error(f"[EMBED] Gemini request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"[EMBED] Gemini error: {e}")
            raise

    @property
    def id(self) -> str:
        """Unique identifier for this provider instance."""
        return f"gemini:{self._model}"

    @property
    def model(self) -> str:
        """Model name used for embeddings."""
        return self._model

    @property
    def vector_dimension(self) -> int:
        """Dimension of the output embedding vectors."""
        return self._dimension
