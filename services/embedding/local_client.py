"""本地嵌入客户端

使用 sentence-transformers 模型在本地生成文本嵌入向量。
"""
from pathlib import Path
from typing import List

from .base import EmbeddingProvider
from config.logging import get_logger


logger = get_logger(__name__)


# Default models
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIMENSION = 384

# Common model dimensions
MODEL_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
}


class LocalEmbeddingClient(EmbeddingProvider):
    """Local embedding client using sentence-transformers.

    Features:
    - No API calls, fully offline
    - Supports any sentence-transformers model
    - Model caching for faster loading
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_dir: str | None = None,
        device: str = "cpu",
    ):
        """Initialize the local embedding client.

        Args:
            model: Model name or path.
            cache_dir: Directory to cache downloaded models.
            device: Device to run on ("cpu", "cuda", "mps").
        """
        self._model_name = model
        self._cache_dir = cache_dir
        self._device = device

        # Get model dimension
        self._dimension = MODEL_DIMENSIONS.get(model, DEFAULT_DIMENSION)

        # Lazy-loaded model
        self._model = None

    @classmethod
    async def create(cls, config) -> "LocalEmbeddingClient":
        """Create a local embedding client from config.

        Args:
            config: Memory search configuration.

        Returns:
            Initialized LocalEmbeddingClient instance.
        """
        model = DEFAULT_MODEL
        cache_dir = None

        # Check if local model path is specified
        if config.local.model_path:
            model_path = Path(config.local.model_path).expanduser()
            if model_path.exists():
                model = str(model_path)
            else:
                logger.warning(f"[EMBED] Model path not found: {model_path}")

        # Check cache dir
        if config.local.model_cache_dir:
            cache_dir = Path(config.local.model_cache_dir).expanduser()

        return cls(
            model=model,
            cache_dir=str(cache_dir) if cache_dir else None,
        )

    def _load_model(self):
        """Load the sentence-transformers model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            )

        logger.info(f"[EMBED] Loading model: {self._model_name}")

        # Prepare cache folder
        cache_folder = self._cache_dir

        # Load model
        self._model = SentenceTransformer(
            self._model_name,
            cache_folder=cache_folder,
            device=self._device,
        )

        # Update dimension from actual model
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"[EMBED] Model loaded, dimension: {self._dimension}")

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

        # Load model if needed
        self._load_model()

        # Run encoding in thread pool to avoid blocking
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor() as executor:
            embeddings = await loop.run_in_executor(
                executor,
                self._model.encode,
                texts,
            )

        # Convert to list of lists
        return embeddings.tolist()

    @property
    def id(self) -> str:
        """Unique identifier for this provider instance."""
        return f"local:{self._model_name}"

    @property
    def model(self) -> str:
        """Model name used for embeddings."""
        return self._model_name

    @property
    def vector_dimension(self) -> int:
        """Dimension of the output embedding vectors."""
        return self._dimension
