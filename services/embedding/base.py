"""嵌入服务提供者基类

定义嵌入服务提供者的抽象接口。
"""
from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in batch.

        Args:
            texts: List of texts to embed.

        Returns:
            A list of embedding vectors.
        """
        pass

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for this provider instance."""
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        """Model name used for embeddings."""
        pass

    @property
    @abstractmethod
    def vector_dimension(self) -> int:
        """Dimension of the output embedding vectors."""
        pass
