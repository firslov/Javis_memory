"""配置管理

从 config/servers.yaml 加载配置，支持 RAG 向量检索记忆系统。
"""
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    base_url: str
    api_key: str
    models: List[str] = Field(default_factory=list)


class DatabaseConfig(BaseModel):
    url: str = "sqlite+aiosqlite:///./javis.db"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    slow_request_threshold: float = 5.0


class CacheConfig(BaseModel):
    enabled: bool = True


# ============================================================================
# RAG-based Memory Search Configuration
# ============================================================================

class RemoteEmbeddingConfig(BaseModel):
    """Remote embedding API configuration."""
    api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    base_url: Optional[str] = None

    class BatchConfig(BaseModel):
        enabled: bool = True
        wait: bool = True
        concurrency: int = 2
        timeout_minutes: int = 60

    batch: BatchConfig = Field(default_factory=BatchConfig)


class LocalEmbeddingConfig(BaseModel):
    """Local embedding model configuration."""
    model_path: str = ""
    model_cache_dir: str = ""
    device: str = "cpu"


class MemoryStoreConfig(BaseModel):
    """Memory storage configuration."""
    path: str = "~/.javis/memory/{user_id}/memory.sqlite"

    class VectorConfig(BaseModel):
        enabled: bool = True
        extension_path: str = ""

    vector: VectorConfig = Field(default_factory=VectorConfig)


class ChunkingConfig(BaseModel):
    """Text chunking configuration."""
    tokens: int = 400
    overlap: int = 80


class SyncConfig(BaseModel):
    """Memory synchronization configuration."""
    on_session_start: bool = True
    on_search: bool = False  # Default to false to avoid conflicts
    watch: bool = True
    watch_debounce_ms: int = 1500
    interval_minutes: int = 0


class QueryConfig(BaseModel):
    """Search query configuration."""
    max_results: int = 6
    min_score: float = 0.35

    class HybridConfig(BaseModel):
        enabled: bool = True
        vector_weight: float = 0.7
        text_weight: float = 0.3
        candidate_multiplier: int = 4

    hybrid: HybridConfig = Field(default_factory=HybridConfig)


class EmbeddingCacheConfig(BaseModel):
    """Embedding cache configuration."""
    enabled: bool = True
    max_entries: int = 10000


# ============================================================================
# Memory Lifecycle Configuration
# ============================================================================

class ImportanceWeightConfig(BaseModel):
    """记忆重要性评分权重配置."""
    novelty: float = 0.25
    sentiment: float = 0.15
    feedback: float = 0.30
    access: float = 0.20
    density: float = 0.10


class MemoryImportanceConfig(BaseModel):
    """记忆重要性评分配置."""
    enabled: bool = True
    weights: ImportanceWeightConfig = Field(default_factory=ImportanceWeightConfig)


class MemoryLifecycleConfig(BaseModel):
    """记忆生命周期管理配置."""
    enabled: bool = True
    working_memory_days: int = 7
    long_term_threshold: float = 0.6
    archive_threshold: float = 0.3
    archive_after_days: int = 90
    promotion_access_count: int = 5

    # 重要性评分配置
    importance: MemoryImportanceConfig = Field(default_factory=MemoryImportanceConfig)


class MemoryConsolidationConfig(BaseModel):
    """记忆合并配置."""
    enabled: bool = True
    similarity_threshold: float = 0.85
    min_cluster_size: int = 3
    time_window_days: int = 7
    schedule: str = "0 2 * * *"  # cron表达式：每天凌晨2点


class MemorySearchConfig(BaseModel):
    """RAG-based memory search configuration."""
    enabled: bool = True
    sources: List[str] = Field(default_factory=lambda: ["memory"])
    extra_paths: List[str] = Field(default_factory=list)
    memory_files_dir: str = "~/.javis/memory/{user_id}"  # 用户记忆文件存储目录

    # Embedding Provider
    provider: str = "auto"  # "openai", "gemini", "local", "auto"
    model: str = ""
    fallback: str = "none"

    # Remote/Local configuration
    remote: Optional[RemoteEmbeddingConfig] = None
    local: LocalEmbeddingConfig = Field(default_factory=LocalEmbeddingConfig)

    # Storage configuration
    store: MemoryStoreConfig = Field(default_factory=MemoryStoreConfig)

    # Chunking configuration
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)

    # Sync configuration
    sync: SyncConfig = Field(default_factory=SyncConfig)

    # Query configuration
    query: QueryConfig = Field(default_factory=QueryConfig)

    # Cache configuration
    cache: EmbeddingCacheConfig = Field(default_factory=EmbeddingCacheConfig)

    # 记忆生命周期管理配置
    lifecycle: MemoryLifecycleConfig = Field(default_factory=MemoryLifecycleConfig)

    # 记忆合并配置
    consolidation: MemoryConsolidationConfig = Field(default_factory=MemoryConsolidationConfig)


class Settings(BaseModel):
    servers: Dict[str, ServerConfig] = Field(default_factory=dict)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    # RAG-based memory search configuration
    memory_search: MemorySearchConfig = Field(default_factory=MemorySearchConfig)

    @classmethod
    def from_yaml(cls, path: str = "config/servers.yaml") -> "Settings":
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        servers = {
            name: ServerConfig(**srv)
            for name, srv in data.get("servers", {}).items()
        }

        return cls(
            servers=servers,
            database=DatabaseConfig(**data.get("database", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            cache=CacheConfig(**data.get("cache", {})),
            memory_search=MemorySearchConfig(**data.get("memory_search", {})),
        )


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_yaml()
    return _settings


def reload_settings(config_path: str = "config/servers.yaml") -> Settings:
    global _settings
    _settings = Settings.from_yaml(config_path)
    return _settings
