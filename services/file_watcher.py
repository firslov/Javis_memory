"""文件监控服务

监控记忆文件并在检测到更改时触发增量同步。
"""
import asyncio
from pathlib import Path
from threading import Thread
from typing import List, Optional, Callable, Any

from config.logging import get_logger


logger = get_logger(__name__)

# Main event loop reference for thread-safe callback scheduling
_main_event_loop: Optional[asyncio.AbstractEventLoop] = None


def set_main_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Set the main event loop for thread-safe operations."""
    global _main_event_loop
    _main_event_loop = loop


class FileWatcher:
    """File watcher using watchdog for monitoring memory files.

    Features:
    - Monitors multiple directories
    - Debouncing to avoid excessive syncs
    - Async callback support
    - Auto-restart on errors
    """

    def __init__(
        self,
        watch_paths: List[str],
        callback: Callable[[], Any],
        debounce_ms: int = 1500,
        patterns: Optional[List[str]] = None,
    ):
        """Initialize the file watcher.

        Args:
            watch_paths: List of directory paths to watch.
            callback: Async callback to invoke on changes.
            debounce_ms: Debounce delay in milliseconds.
            patterns: File patterns to watch (default: *.md).
        """
        self.watch_paths = [Path(p).expanduser() for p in watch_paths]
        self.callback = callback
        self.debounce_ms = debounce_ms
        self.patterns = patterns or ["*.md"]

        # Watchdog observer
        self._observer = None
        self._task: Optional[asyncio.Task] = None
        self._debounce_task: Optional[asyncio.Task] = None

        # State
        self._running = False
        self._pending_changes = False

    async def start(self) -> None:
        """Start the file watcher.

        Creates observer and starts watching configured paths.
        """
        if self._running:
            logger.warning("[WATCH] Already running")
            return

        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
        except ImportError:
            logger.warning(
                "[WATCH] watchdog not installed. Install with: pip install watchdog"
            )
            return

        # Create event handler
        class Handler(FileSystemEventHandler):
            def __init__(self, watcher: FileWatcher):
                self.watcher = watcher

            def on_modified(self, event):
                if not event.is_directory:
                    self.watcher._schedule_debounce()

            def on_created(self, event):
                if not event.is_directory:
                    self.watcher._schedule_debounce()

            def on_deleted(self, event):
                if not event.is_directory:
                    self.watcher._schedule_debounce()

        # Create and start observer
        self._observer = Observer()
        handler = Handler(self)

        for watch_path in self.watch_paths:
            if not watch_path.exists():
                logger.warning(f"[WATCH] Path does not exist: {watch_path}")
                continue

            self._observer.schedule(
                handler,
                str(watch_path),
                recursive=True,
            )
            logger.info(f"[WATCH] Monitoring: {watch_path}")

        self._observer.start()
        self._running = True

        logger.info(f"[WATCH] Started (debounce: {self.debounce_ms}ms)")

    async def stop(self) -> None:
        """Stop the file watcher."""
        if not self._running:
            return

        self._running = False

        # Cancel debounce task
        if self._debounce_task:
            self._debounce_task.cancel()
            self._debounce_task = None

        # Stop observer
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None

        logger.info("[WATCH] Stopped")

    def _schedule_debounce(self) -> None:
        """Schedule a debounced sync."""
        self._pending_changes = True

        if self._debounce_task and not self._debounce_task.done():
            # Cancel existing task
            self._debounce_task.cancel()

        # Schedule new task (thread-safe)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, use main event loop
            loop = _main_event_loop
            if loop is None:
                logger.warning("[WATCH] No event loop for callback")
                return

        if loop is None or loop.is_closed():
            logger.warning("[WATCH] Event loop unavailable")
            return

        self._debounce_task = asyncio.run_coroutine_threadsafe(
            self._debounced_callback(), loop
        )

    async def _debounced_callback(self) -> None:
        """Debounced callback invocation."""
        try:
            await asyncio.sleep(self.debounce_ms / 1000.0)

            if self._pending_changes and self._running:
                self._pending_changes = False
                logger.debug("[WATCH] Triggering sync callback")
                await self.callback()

        except asyncio.CancelledError:
            # Task was cancelled, ignore
            pass
        except Exception as e:
            logger.error(f"[WATCH] Callback error: {e}")

    @property
    def is_running(self) -> bool:
        """Check if the file watcher is running."""
        return self._running


class FileWatcherManager:
    """Manager for multiple file watchers.

    Handles lifecycle and provides cleanup.
    """

    def __init__(self):
        """Initialize the manager."""
        self._watchers: List[FileWatcher] = []

    async def create_watcher(
        self,
        watch_paths: List[str],
        callback: Callable[[], Any],
        debounce_ms: int = 1500,
    ) -> FileWatcher:
        """Create and start a new file watcher.

        Args:
            watch_paths: List of directory paths to watch.
            callback: Async callback to invoke on changes.
            debounce_ms: Debounce delay in milliseconds.

        Returns:
            FileWatcher instance.
        """
        watcher = FileWatcher(
            watch_paths=watch_paths,
            callback=callback,
            debounce_ms=debounce_ms,
        )
        await watcher.start()
        self._watchers.append(watcher)
        return watcher

    async def stop_all(self) -> None:
        """Stop all file watchers."""
        for watcher in self._watchers:
            await watcher.stop()
        self._watchers.clear()


# Global manager instance
_manager: Optional[FileWatcherManager] = None


def get_file_watcher_manager() -> FileWatcherManager:
    """Get the global file watcher manager.

    Returns:
        FileWatcherManager instance.
    """
    global _manager
    if _manager is None:
        _manager = FileWatcherManager()
    return _manager


async def start_memory_file_watcher(
    config: Any,  # MemorySearchConfig
    sync_callback: Callable[[], Any],
) -> Optional[FileWatcher]:
    """Start file watcher for memory sources.

    Args:
        config: Memory search configuration.
        sync_callback: Callback to invoke for syncing.

    Returns:
        FileWatcher instance or None if not configured.
    """
    if not config.sync.watch:
        return None

    # Collect watch paths
    watch_paths = []

    if "memory" in config.sources:
        # Watch user memory directory parent (~/javis/memory) to monitor all users
        # Extract parent directory from the user-specific path template
        memory_files_dir = Path(config.memory_files_dir.format(user_id=1)).expanduser()
        memory_parent_dir = memory_files_dir.parent
        memory_parent_dir.mkdir(parents=True, exist_ok=True)
        watch_paths.append(str(memory_parent_dir))

        # Also watch legacy memory directory for backward compatibility
        legacy_memory_dir = Path("memory").expanduser()
        if legacy_memory_dir.exists():
            watch_paths.append(str(legacy_memory_dir))

        # Watch current directory for MEMORY.md (legacy)
        watch_paths.append(str(Path.cwd()))

    if "sessions" in config.sources:
        # Watch sessions directory
        sessions_dir = Path("~/.ai-agent/sessions").expanduser()
        if sessions_dir.exists():
            watch_paths.append(str(sessions_dir))

    # Watch extra paths
    for extra_path in config.extra_paths:
        extra = Path(extra_path).expanduser()
        if extra.exists():
            if extra.is_dir():
                watch_paths.append(str(extra))
            else:
                # Watch parent directory for single file
                watch_paths.append(str(extra.parent))

    if not watch_paths:
        logger.warning("[WATCH] No paths to watch")
        return None

    manager = get_file_watcher_manager()
    return await manager.create_watcher(
        watch_paths=watch_paths,
        callback=sync_callback,
        debounce_ms=config.sync.watch_debounce_ms,
    )


async def stop_all_file_watchers() -> None:
    """Stop all file watchers."""
    global _manager
    if _manager:
        await _manager.stop_all()
        _manager = None
