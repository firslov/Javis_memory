"""嵌入服务

支持多种嵌入服务提供者：
    - OpenAI 兼容 API（OpenAI、DeepSeek、豆包等）
    - Gemini API
    - 本地 sentence-transformers 模型
"""

from .base import EmbeddingProvider
from .factory import create_embedding_provider

__all__ = ["EmbeddingProvider", "create_embedding_provider"]
