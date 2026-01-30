"""记忆索引管理器

管理 RAG 向量检索记忆系统的文件扫描、分块、嵌入和检索。
"""
import asyncio
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import aiosqlite
from aiosqlite import Connection

from .chunker import Chunker
from .embedding import create_embedding_provider, EmbeddingProvider
from .search_engine import SearchEngine, SearchResult
from config.logging import get_logger


logger = get_logger(__name__)

# Global lock for database access per user
_db_locks: Dict[int, asyncio.Lock] = {}


def _get_db_lock(user_id: int) -> asyncio.Lock:
    """Get or create a lock for a user's database access."""
    if user_id not in _db_locks:
        _db_locks[user_id] = asyncio.Lock()
    return _db_locks[user_id]


class MemoryIndexManager:
    """Manager for memory indexing and retrieval.

    Responsibilities:
    - Scan memory files and detect changes
    - Trigger chunking and embedding
    - Manage index database
    - Perform hybrid search
    - Safe file reading
    """

    # Class variable to track one-time logs
    _initialized_users = set()

    def __init__(
        self,
        user_id: int,
        db_path: str,
        config: Any,  # MemorySearchConfig
    ):
        """Initialize the memory index manager.

        Args:
            user_id: User ID for this index.
            db_path: Path to the index database.
            config: Memory search configuration.
        """
        self.user_id = user_id
        self.db_path = Path(db_path).expanduser()
        self.config = config

        # Services
        self.chunker: Optional[Chunker] = None
        self.embedding_provider: Optional[EmbeddingProvider] = None
        self.search_engine: Optional[SearchEngine] = None

        # Database connection
        self._db: Optional[Connection] = None

        # Sync state
        self._syncing = False
        self._sync_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize the index manager.

        Creates database, initializes services, etc.
        """
        import aiosqlite

        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create connection with WAL mode for better concurrency
        # Note: check_same_thread=False is needed because the connection may be
        # used in different threads (e.g., file watcher vs request handler)
        self._db = await aiosqlite.connect(
            str(self.db_path),
            check_same_thread=False
        )

        # Enable WAL mode for better concurrent access
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")  # 5 second timeout

        # Load sqlite-vec extension
        try:
            import sqlite_vec
            # Enable loading extensions
            await self._db.enable_load_extension(True)
            # Load sqlite-vec extension
            await self._db.load_extension(sqlite_vec.loadable_path())
        except Exception as e:
            raise RuntimeError(
                f"sqlite-vec extension is required but not available. "
                f"Install with: pip install 'sqlite-vec>=0.1.0'. Error: {e}"
            ) from e

        await self._db.commit()

        # Initialize database schema
        await self._init_db()

        # Initialize services
        await self._init_services()

        logger.info(f"[MEMORY] Index manager initialized for user {self.user_id}")

    async def close(self) -> None:
        """Close the index manager and database connection."""
        if self._db:
            await self._db.close()
            self._db = None

        if self.embedding_provider:
            await self.embedding_provider.close()
            self.embedding_provider = None

        logger.info(f"[MEMORY] Index manager closed for user {self.user_id}")

    async def _init_db(self) -> None:
        """Initialize database schema."""
        # Create meta table for configuration tracking (from design doc)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS memory_index_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # Create tables
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS memory_files (
                user_id INTEGER NOT NULL,
                path TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'memory',
                hash TEXT NOT NULL,
                mtime INTEGER NOT NULL,
                size INTEGER NOT NULL,
                indexed_at TIMESTAMP,
                chunk_count INTEGER DEFAULT 0,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, path)
            )
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS memory_chunks (
                id TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                path TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'memory',
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                hash TEXT NOT NULL,
                text TEXT NOT NULL,
                model TEXT NOT NULL,
                provider TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                embedding TEXT NOT NULL,
                token_count INTEGER DEFAULT 0,
                extra_metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id, user_id)
            )
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                hash TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                embedding TEXT NOT NULL,
                hit_count INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (provider, model, hash)
            )
        """)

        # Create indexes
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS ix_memory_chunks_user_path
            ON memory_chunks(user_id, path)
        """)

        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS ix_memory_chunks_source
            ON memory_chunks(user_id, source)
        """)

        # Create index for cache LRU pruning
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS ix_embedding_cache_updated_at
            ON embedding_cache(updated_at)
        """)

        # Note: Vector table (memory_chunks_vec) is NOT created here
        # It will be created in _init_services() after we know the embedding dimension

        # Try to create FTS5 index (optional)
        try:
            await self._db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_chunks_fts
                USING fts5(text, content='memory_chunks', content_rowid='rowid')
            """)
            await self._db.execute("""
                CREATE TRIGGER IF NOT EXISTS memory_chunks_fts_insert
                AFTER INSERT ON memory_chunks BEGIN
                    INSERT INTO memory_chunks_fts(rowid, text)
                    VALUES (NEW.rowid, NEW.text);
                END
            """)
            await self._db.execute("""
                CREATE TRIGGER IF NOT EXISTS memory_chunks_fts_delete
                AFTER DELETE ON memory_chunks BEGIN
                    DELETE FROM memory_chunks_fts WHERE rowid = OLD.rowid;
                END
            """)
        except Exception as e:
            logger.debug(f"[MEMORY] FTS5 not available: {e}")

        await self._db.commit()

    async def _init_services(self) -> None:
        """Initialize chunker, embedding provider, and search engine."""
        # Initialize chunker
        self.chunker = Chunker(
            tokens=self.config.chunking.tokens,
            overlap=self.config.chunking.overlap,
        )

        # Initialize embedding provider
        self.embedding_provider = await create_embedding_provider(self.config)

        # Initialize search engine
        self.search_engine = SearchEngine(
            vector_weight=self.config.query.hybrid.vector_weight,
            keyword_weight=self.config.query.hybrid.text_weight,
            candidate_multiplier=self.config.query.hybrid.candidate_multiplier,
        )

        # Create vector index table with correct dimension (now that we know it)
        dimension = self._get_vector_dimension()

        try:
            # Check if vector table exists and get its dimension
            cursor = await self._db.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='memory_chunks_vec'"
            )
            row = await cursor.fetchone()
            await cursor.close()

            # Parse existing dimension from schema if table exists
            existing_dimension = None
            if row:
                # Schema format: "...embedding FLOAT[dimension]..."
                import re
                match = re.search(r'embedding\s+FLOAT\[(\d+)\]', row[0])
                if match:
                    existing_dimension = int(match.group(1))

            # Only recreate if dimension changed or table doesn't exist
            if existing_dimension != dimension:
                if existing_dimension:
                    logger.info(f"[MEMORY] Recreating vector table: dimension {existing_dimension} → {dimension}")
                else:
                    logger.info(f"[MEMORY] Creating vector table with dimension {dimension}")

                await self._db.execute("DROP TABLE IF EXISTS memory_chunks_vec")
                await self._db.execute(f"""
                    CREATE VIRTUAL TABLE memory_chunks_vec
                    USING vec0(id TEXT, user_id INTEGER, embedding FLOAT[{dimension}])
                """)
                await self._db.commit()

                # Re-index all chunks if we recreated the table
                if existing_dimension is not None:  # Only reindex if table existed before
                    logger.info(f"[MEMORY] Vector table recreated, re-indexing chunks...")
                    await self._reindex_all_chunks()
            else:
                logger.debug(f"[MEMORY] Vector table exists with dimension {dimension}")

        except Exception as e:
            logger.error(f"[MEMORY] Failed to create vector table: {e}")
            raise

        # Only log initialization once per user
        if self.user_id not in self._initialized_users:
            logger.debug(
                f"[MEMORY] Services ready for user {self.user_id}: "
                f"chunker={self.chunker.tokens} tokens, "
                f"embedding={self.embedding_provider.id}"
            )
            self._initialized_users.add(self.user_id)

    def _get_vector_dimension(self) -> int:
        """Get the vector dimension for the configured model."""
        if self.embedding_provider:
            return self.embedding_provider.vector_dimension
        return 1536  # Default for OpenAI embeddings

    async def sync(self, force: bool = False) -> Dict[str, Any]:
        """Synchronize the memory index.

        Scans memory files, detects changes, and updates the index.

        Args:
            force: Force full re-index.

        Returns:
            Sync statistics.
        """
        lock = _get_db_lock(self.user_id)

        if self._syncing:
            logger.debug("[MEMORY] Sync already in progress, skipping")
            return {"status": "skipped", "reason": "already_syncing"}

        self._syncing = True

        try:
            # Acquire lock to prevent concurrent access
            async with lock:
                # Check if full reindex is needed (from design doc)
                needs_full_reindex = force or await self._needs_reindex()

                stats = {
                    "status": "success",
                    "files_scanned": 0,
                    "files_updated": 0,
                    "chunks_added": 0,
                    "chunks_removed": 0,
                    "errors": [],
                    "full_reindex": needs_full_reindex,
                }

                # Collect memory source paths
                source_paths = self._collect_source_paths()

                for source, paths in source_paths.items():
                    for path in paths:
                        try:
                            file_stats = await self._sync_file(path, source, needs_full_reindex)
                            stats["files_scanned"] += 1
                            if file_stats["updated"]:
                                stats["files_updated"] += 1
                                stats["chunks_added"] += file_stats["chunks_added"]
                                stats["chunks_removed"] += file_stats["chunks_removed"]

                        except Exception as e:
                            logger.error(f"[MEMORY] Error syncing {path}: {e}")
                            stats["errors"].append(f"{path}: {str(e)}")

                # Update metadata after successful sync
                if stats["files_updated"] > 0:
                    await self._update_meta()

                # Only log if something changed or there were errors
                if stats["files_updated"] > 0 or stats["errors"]:
                    logger.debug(f"[MEMORY] Sync completed: {stats}")
                return stats

        finally:
            self._syncing = False
            self._sync_event.set()
            self._sync_event.clear()

    def _collect_source_paths(self) -> Dict[str, List[Path]]:
        """Collect all memory source file paths.

        Returns:
            Dict mapping source type to list of paths.
        """
        sources = {
            "memory": [],
            "sessions": [],
            "custom": [],
        }

        # Memory directory - use user-specific memory directory
        if "memory" in self.config.sources:
            # Use configured user memory directory
            memory_dir = Path(self.config.memory_files_dir.format(user_id=self.user_id)).expanduser()
            if memory_dir.exists():
                sources["memory"] = list(memory_dir.glob("**/*.md"))

            # Also check legacy memory/ directory for backward compatibility
            legacy_memory_dir = Path("memory").expanduser()
            if legacy_memory_dir.exists():
                sources["memory"].extend(legacy_memory_dir.glob("**/*.md"))

            # Check for MEMORY.md in current directory (legacy)
            memory_file = Path("MEMORY.md").expanduser()
            if memory_file.exists():
                sources["memory"].append(memory_file)

        # Sessions directory
        if "sessions" in self.config.sources:
            sessions_dir = Path("~/.ai-agent/sessions").expanduser()
            if sessions_dir.exists():
                sources["sessions"] = list(sessions_dir.glob("**/*.md"))

        # Custom extra paths
        for extra_path in self.config.extra_paths:
            extra = Path(extra_path).expanduser()
            if extra.exists():
                if extra.is_dir():
                    sources["custom"].extend(extra.glob("**/*.md"))
                else:
                    sources["custom"].append(extra)

        return sources

    async def _sync_file(
        self,
        file_path: Path,
        source: str,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Sync a single file.

        Args:
            file_path: Path to the file.
            source: Source type (memory, sessions, custom).
            force: Force re-indexing.

        Returns:
            File sync statistics.
        """
        stats = {
            "updated": False,
            "chunks_added": 0,
            "chunks_removed": 0,
        }

        # Ensure services are initialized
        if self.chunker is None:
            await self._init_services()

        # Get file info
        try:
            stat = file_path.stat()
            file_hash = self._calculate_file_hash(file_path)
        except Exception as e:
            logger.error(f"[MEMORY] Failed to stat file {file_path}: {e}")
            return stats

        # Check if update is needed
        rel_path = str(file_path)
        if not force:
            cursor = await self._db.execute(
                "SELECT hash FROM memory_files WHERE user_id = ? AND path = ?",
                (self.user_id, rel_path)
            )
            row = await cursor.fetchone()
            await cursor.close()

            if row and row[0] == file_hash:
                # File unchanged
                return stats

        # Read and chunk file
        chunks = self.chunker.chunk_file(file_path)
        if not chunks:
            logger.warning(f"[MEMORY] No chunks generated for {file_path}")
            return stats

        # Remove old chunks for this file (both main table and vector index)
        # First, get the IDs to delete from vector index
        await self._db.execute(
            "DELETE FROM memory_chunks_vec WHERE user_id = ? AND id IN (SELECT id FROM memory_chunks WHERE user_id = ? AND path = ?)",
            (self.user_id, self.user_id, rel_path)
        )
        # Then delete from main table
        await self._db.execute(
            "DELETE FROM memory_chunks WHERE user_id = ? AND path = ?",
            (self.user_id, rel_path)
        )
        cursor = await self._db.execute(
            "SELECT changes()"
        )
        deleted = await cursor.fetchone()
        await cursor.close()
        stats["chunks_removed"] = deleted[0] if deleted else 0

        # Embed chunks and insert
        for chunk in chunks:
            await self._index_chunk(chunk, rel_path, source)

        stats["updated"] = True
        stats["chunks_added"] = len(chunks)

        # Update file record
        await self._db.execute("""
            INSERT OR REPLACE INTO memory_files
            (user_id, path, source, hash, mtime, size, indexed_at, chunk_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.user_id,
            rel_path,
            source,
            file_hash,
            int(stat.st_mtime),
            stat.st_size,
            datetime.utcnow().isoformat(),
            len(chunks),
        ))

        await self._db.commit()

        return stats

    async def _index_chunk(
        self,
        chunk,
        path: str,
        source: str,
    ) -> None:
        """Index a single text chunk.

        Args:
            chunk: TextChunk object.
            path: Relative file path.
            source: Source type.
        """
        # Check embedding cache
        embedding = await self._get_cached_embedding(chunk.hash)

        if embedding is None:
            # Compute embedding
            embeddings = await self.embedding_provider.embed_batch([chunk.text])
            embedding = embeddings[0]

            # Cache the result
            await self._cache_embedding(chunk.hash, embedding)

        # Insert chunk
        chunk_id = f"{self.user_id}:{chunk.hash}"

        await self._db.execute("""
            INSERT OR REPLACE INTO memory_chunks
            (id, user_id, path, source, start_line, end_line, hash, text,
             model, provider, dimension, embedding, token_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk_id,
            self.user_id,
            path,
            source,
            chunk.start_line,
            chunk.end_line,
            chunk.hash,
            chunk.text,
            self.embedding_provider.model,
            self.embedding_provider.id.split(":")[0],
            self.embedding_provider.vector_dimension,
            json.dumps(embedding),
            chunk.token_count,
        ))

        # Insert into vector index
        # Normalize embedding before storing for better distance-based similarity
        import sqlite_vec
        import math

        # Normalize embedding to unit length
        norm = math.sqrt(sum(v * v for v in embedding))
        if norm > 0:
            normalized_embedding = [v / norm for v in embedding]
        else:
            normalized_embedding = embedding

        embedding_bytes = sqlite_vec.serialize_float32(normalized_embedding)
        await self._db.execute("""
            INSERT OR REPLACE INTO memory_chunks_vec(id, user_id, embedding)
            VALUES (?, ?, ?)
        """, (chunk_id, self.user_id, embedding_bytes))

    async def _get_cached_embedding(self, text_hash: str) -> Optional[List[float]]:
        """Get embedding from cache.

        Args:
            text_hash: SHA-256 hash of the text.

        Returns:
            Cached embedding or None.
        """
        if not self.config.cache.enabled:
            return None

        cursor = await self._db.execute("""
            SELECT embedding FROM embedding_cache
            WHERE provider = ? AND model = ? AND hash = ?
        """, (
            self.embedding_provider.id.split(":")[0],
            self.embedding_provider.model,
            text_hash,
        ))

        row = await cursor.fetchone()
        await cursor.close()

        if row:
            # Update hit count
            await self._db.execute("""
                UPDATE embedding_cache
                SET hit_count = hit_count + 1, updated_at = CURRENT_TIMESTAMP
                WHERE provider = ? AND model = ? AND hash = ?
            """, (
                self.embedding_provider.id.split(":")[0],
                self.embedding_provider.model,
                text_hash,
            ))
            await self._db.commit()

            return json.loads(row[0])

        return None

    async def _cache_embedding(self, text_hash: str, embedding: List[float]) -> None:
        """Cache an embedding result.

        Args:
            text_hash: SHA-256 hash of the text.
            embedding: Embedding vector.
        """
        if not self.config.cache.enabled:
            return

        await self._db.execute("""
            INSERT OR REPLACE INTO embedding_cache
            (provider, model, hash, dimension, embedding)
            VALUES (?, ?, ?, ?, ?)
        """, (
            self.embedding_provider.id.split(":")[0],
            self.embedding_provider.model,
            text_hash,
            len(embedding),
            json.dumps(embedding),
        ))

        # Prune cache if exceeding max entries (LRU by updated_at)
        if self.config.cache.max_entries:
            await self._prune_cache(self.config.cache.max_entries)

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file.

        Args:
            file_path: Path to the file.

        Returns:
            Hexadecimal hash string.
        """
        sha256 = hashlib.sha256()

        # Read file in binary mode
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        return sha256.hexdigest()

    async def search(
        self,
        query: str,
        max_results: int = None,
        min_score: float = None,
    ) -> List[SearchResult]:
        """Search for relevant memories.

        Args:
            query: Search query.
            max_results: Maximum results (from config if not specified).
            min_score: Minimum score threshold (from config if not specified).

        Returns:
            List of search results.
        """
        if max_results is None:
            max_results = self.config.query.max_results
        if min_score is None:
            min_score = self.config.query.min_score

        lock = _get_db_lock(self.user_id)

        # Wait for any ongoing sync to complete
        if lock.locked():
            logger.debug("[MEMORY] Sync in progress, waiting...")
            async with lock:
                pass  # Just wait for lock to be released

        # Sync if configured (but only if not already syncing)
        if self.config.sync.on_search and not self._syncing:
            await self.sync()

        # Check if there are chunks in the database
        cursor = await self._db.execute("SELECT COUNT(*) FROM memory_chunks WHERE user_id = ?", (self.user_id,))
        chunk_count = await cursor.fetchone()
        await cursor.close()

        # Check if there are vectors in the vector table
        cursor = await self._db.execute("SELECT COUNT(*) FROM memory_chunks_vec WHERE user_id = ?", (self.user_id,))
        vec_count = await cursor.fetchone()
        await cursor.close()

        if vec_count[0] == 0:
            logger.warning(f"[MEMORY] No indexed memories for user {self.user_id}. Run sync first.")

        # Embed query
        query_embedding = await self.embedding_provider.embed_query(query)

        # Perform hybrid search with lock to prevent concurrent access
        async with lock:
            results = await self.search_engine.search(
                query_embedding=query_embedding,
                query_text=query,
                user_id=self.user_id,
                max_results=max_results,
                min_score=min_score,
                db_conn=self._db,
            )

        if results:
            logger.info(f"[SEARCH] Found {len(results)} results for: '{query[:50]}...'")
        else:
            logger.debug(f"[SEARCH] No results for: '{query[:50]}...'")

        return results

    async def read_file(
        self,
        rel_path: str,
        from_line: int = None,
        lines: int = None,
    ) -> str:
        """Safely read a file from the memory sources.

        Args:
            rel_path: Relative path to the file.
            from_line: Starting line number.
            lines: Number of lines to read.

        Returns:
            File content.
        """
        # Resolve path
        file_path = self._resolve_path(rel_path)

        if not file_path or not file_path.exists():
            return f"File not found: {rel_path}"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if from_line is None:
                    return f.read()

                # Read specific lines
                all_lines = f.readlines()
                start = max(0, from_line - 1)
                end = len(all_lines) if lines is None else min(start + lines, len(all_lines))
                return "".join(all_lines[start:end])

        except Exception as e:
            logger.error(f"[MEMORY] Error reading {rel_path}: {e}")
            return f"Error reading file: {str(e)}"

    def _resolve_path(self, rel_path: str) -> Optional[Path]:
        """Resolve a relative path to an absolute path.

        Args:
            rel_path: Relative path from memory sources.

        Returns:
            Absolute path or None if not found.
        """
        # Try direct path
        path = Path(rel_path).expanduser()
        if path.exists():
            return path

        # Try user-specific memory directory
        user_memory_dir = Path(self.config.memory_files_dir.format(user_id=self.user_id)).expanduser()
        user_memory_path = user_memory_dir / rel_path
        if user_memory_path.exists():
            return user_memory_path

        # Try legacy memory directory
        memory_path = Path("memory") / rel_path
        if memory_path.exists():
            return memory_path

        # Try sessions directory
        sessions_path = Path("~/.ai-agent/sessions") / rel_path
        if sessions_path.expanduser().exists():
            return sessions_path.expanduser()

        # Try extra paths
        for extra in self.config.extra_paths:
            extra_path = Path(extra).expanduser() / rel_path
            if extra_path.exists():
                return extra_path

        return None

    async def _reindex_all_chunks(self) -> None:
        """Re-index all chunks into the vector table after table recreation."""
        # Get all chunks from the main table
        cursor = await self._db.execute("""
            SELECT id, user_id, path, source, start_line, end_line, hash, text,
                   model, provider, dimension, embedding, token_count
            FROM memory_chunks
            WHERE user_id = ?
        """, (self.user_id,))
        rows = await cursor.fetchall()
        await cursor.close()

        import sqlite_vec
        import math

        # Re-insert each chunk's vector into the vector table
        for row in rows:
            chunk_id = row[0]
            # Column indices: id=0, user_id=1, path=2, source=3, start_line=4, end_line=5,
            #                  hash=6, text=7, model=8, provider=9, dimension=10, embedding=11, token_count=12
            embedding_data = row[11]

            # Handle both binary (new) and JSON (old) formats
            if isinstance(embedding_data, bytes):
                # Binary format: deserialize float32 array
                embedding = sqlite_vec.deserialize_float32(embedding_data)
            else:
                # JSON format (old): parse JSON string
                import json
                embedding = json.loads(embedding_data)

            # Normalize to unit length
            norm = math.sqrt(sum(v * v for v in embedding))
            if norm > 0:
                embedding = [v / norm for v in embedding]

            embedding_bytes = sqlite_vec.serialize_float32(embedding)

            await self._db.execute("""
                INSERT OR REPLACE INTO memory_chunks_vec(id, user_id, embedding)
                VALUES (?, ?, ?)
            """, (chunk_id, self.user_id, embedding_bytes))

        await self._db.commit()
        logger.info(f"[MEMORY] Re-indexed {len(rows)} chunks")

    async def _prune_cache(self, max_entries: int) -> None:
        """Prune embedding cache to max_entries using LRU by updated_at.

        From design doc: delete oldest entries when exceeding limit.
        """
        cursor = await self._db.execute("SELECT COUNT(*) as c FROM embedding_cache")
        row = await cursor.fetchone()
        await cursor.close()

        count = row[0] if row else 0
        if count <= max_entries:
            return

        excess = count - max_entries
        await self._db.execute("""
            DELETE FROM embedding_cache
            WHERE rowid IN (
                SELECT rowid FROM embedding_cache
                ORDER BY updated_at ASC
                LIMIT ?
            )
        """, (excess,))
        await self._db.commit()
        logger.debug(f"[MEMORY] Pruned {excess} cache entries")

    def _get_provider_key(self) -> str:
        """Compute provider key excluding authorization headers.

        From design doc: exclude sensitive headers from cache key to allow
        cache sharing across different API keys with same model configuration.
        """
        import hashlib
        import json

        if self.embedding_provider.id == "openai":
            # For OpenAI, use base_url and model (exclude api_key)
            key_data = {
                "provider": "openai",
                "model": self.embedding_provider.model,
            }
        elif self.embedding_provider.id == "gemini":
            key_data = {
                "provider": "gemini",
                "model": self.embedding_provider.model,
            }
        else:
            key_data = {
                "provider": self.embedding_provider.id,
                "model": self.embedding_provider.model,
            }

        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    async def _read_meta(self) -> Optional[dict]:
        """Read index metadata from database."""
        cursor = await self._db.execute(
            "SELECT value FROM memory_index_meta WHERE key = ?",
            ("memory_index_meta_v1",)
        )
        row = await cursor.fetchone()
        await cursor.close()

        if row:
            import json
            return json.loads(row[0])
        return None

    async def _write_meta(self, meta: dict) -> None:
        """Write index metadata to database."""
        import json
        await self._db.execute("""
            INSERT OR REPLACE INTO memory_index_meta (key, value)
            VALUES (?, ?)
        """, ("memory_index_meta_v1", json.dumps(meta)))
        await self._db.commit()

    async def _needs_reindex(self) -> bool:
        """Check if full reindex is needed based on metadata.

        From design doc: reindex when model, provider, or chunking config changes.
        """
        meta = await self._read_meta()
        if not meta:
            return True  # No metadata, need first index

        # Check if configuration changed
        current_meta = {
            "model": self.embedding_provider.model,
            "provider": self.embedding_provider.id,
            "chunkTokens": self.chunker.tokens if self.chunker else 400,
            "chunkOverlap": self.chunker.overlap if self.chunker else 80,
            "vectorDims": self.embedding_provider.vector_dimension,
        }

        # Check for differences
        for key, value in current_meta.items():
            if meta.get(key) != value:
                logger.info(f"[MEMORY] Config changed: {key} {meta.get(key)} -> {value}, reindex needed")
                return True

        return False

    async def _update_meta(self) -> None:
        """Update metadata after successful index."""
        meta = {
            "model": self.embedding_provider.model,
            "provider": self.embedding_provider.id,
            "chunkTokens": self.chunker.tokens if self.chunker else 400,
            "chunkOverlap": self.chunker.overlap if self.chunker else 80,
            "vectorDims": self.embedding_provider.vector_dimension,
        }
        await self._write_meta(meta)

    def status(self) -> Dict[str, Any]:
        """Get system status.

        Returns:
            Status information.
        """
        return {
            "user_id": self.user_id,
            "db_path": str(self.db_path),
            "chunker_tokens": self.chunker.tokens if self.chunker else None,
            "embedding_provider": self.embedding_provider.id if self.embedding_provider else None,
            "embedding_dimension": self.embedding_provider.vector_dimension if self.embedding_provider else None,
            "syncing": self._syncing,
        }
