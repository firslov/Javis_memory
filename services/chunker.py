"""文本分块服务

将 Markdown 文件按 token 数分块，支持块间重叠。
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from config.logging import get_logger


logger = get_logger(__name__)


@dataclass
class TextChunk:
    """A text chunk with metadata."""

    text: str
    start_line: int
    end_line: int
    hash: str
    token_count: int


class Chunker:
    """Text chunker for splitting markdown files into token-based chunks."""

    # Default chunking parameters
    DEFAULT_TOKENS = 400
    DEFAULT_OVERLAP = 80

    def __init__(
        self,
        tokens: int = DEFAULT_TOKENS,
        overlap: int = DEFAULT_OVERLAP,
    ):
        """Initialize the chunker.

        Args:
            tokens: Target tokens per chunk.
            overlap: Overlap tokens between chunks.
        """
        self.tokens = tokens
        self.overlap = overlap

    def chunk_text(
        self,
        text: str,
        start_line: int = 0,
    ) -> List[TextChunk]:
        """Split text into chunks.

        Args:
            text: The text to chunk.
            start_line: Starting line number.

        Returns:
            List of TextChunk objects.
        """
        if not text:
            return []

        # Split into lines
        lines = text.splitlines(keepends=True)

        chunks = []
        current_chunk_lines = []
        current_tokens = 0
        chunk_start_line = start_line

        for i, line in enumerate(lines):
            line_tokens = self._estimate_tokens(line)

            # Check if adding this line would exceed chunk size
            if current_tokens + line_tokens > self.tokens and current_chunk_lines:
                # Save current chunk
                chunk_text = "".join(current_chunk_lines)
                chunks.append(self._create_chunk(
                    chunk_text,
                    chunk_start_line,
                    chunk_start_line + len(current_chunk_lines),
                ))

                # Start new chunk with overlap
                overlap_lines = self._get_overlap_lines(current_chunk_lines)
                current_chunk_lines = overlap_lines
                current_tokens = sum(
                    self._estimate_tokens(line) for line in overlap_lines
                )
                chunk_start_line = start_line + i - len(overlap_lines)

            # Add line to current chunk
            current_chunk_lines.append(line)
            current_tokens += line_tokens

        # Add final chunk
        if current_chunk_lines:
            chunk_text = "".join(current_chunk_lines)
            chunks.append(self._create_chunk(
                chunk_text,
                chunk_start_line,
                chunk_start_line + len(current_chunk_lines),
            ))

        logger.debug(f"[CHUNK] Created {len(chunks)} chunks")
        return chunks

    def chunk_file(
        self,
        file_path: str | Path,
        encoding: str = "utf-8",
    ) -> List[TextChunk]:
        """Chunk a markdown file.

        Args:
            file_path: Path to the file.
            encoding: File encoding.

        Returns:
            List of TextChunk objects.
        """
        file_path = Path(file_path)

        try:
            with open(file_path, "r", encoding=encoding) as f:
                text = f.read()

            return self.chunk_text(text, start_line=0)

        except FileNotFoundError:
            logger.warning(f"[CHUNK] File not found: {file_path}")
            return []
        except UnicodeDecodeError:
            logger.warning(f"[CHUNK] Decode failed: {file_path}")
            return []

    def _create_chunk(
        self,
        text: str,
        start_line: int,
        end_line: int,
    ) -> TextChunk:
        """Create a TextChunk object.

        Args:
            text: Chunk text.
            start_line: Starting line number.
            end_line: Ending line number.

        Returns:
            TextChunk object.
        """
        # Strip leading/trailing whitespace
        text = text.strip()

        # Calculate hash
        chunk_hash = self._calculate_hash(text)

        # Estimate token count
        token_count = self._estimate_tokens(text)

        return TextChunk(
            text=text,
            start_line=start_line,
            end_line=end_line,
            hash=chunk_hash,
            token_count=token_count,
        )

    def _calculate_hash(self, text: str) -> str:
        """Calculate SHA-256 hash of text.

        Args:
            text: Text to hash.

        Returns:
            Hexadecimal hash string.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a simple heuristic: ~4 characters per token for English,
        ~2 characters per token for Chinese.

        Args:
            text: Text to estimate.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0

        # Count Chinese characters
        chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        # Count other characters
        other_chars = len(text) - chinese_chars

        # Estimate tokens (Chinese ~2 chars/token, English ~4 chars/token)
        estimated = (chinese_chars / 2) + (other_chars / 4)

        return int(estimated) + 1  # At least 1 token

    def _get_overlap_lines(self, lines: List[str]) -> List[str]:
        """Get overlap lines from previous chunk.

        Args:
            lines: Lines from previous chunk.

        Returns:
            List of lines to overlap.
        """
        if not lines:
            return []

        overlap_tokens = self.overlap
        result = []
        current_tokens = 0

        # Take lines from the end until we reach overlap tokens
        for line in reversed(lines):
            line_tokens = self._estimate_tokens(line)
            if current_tokens + line_tokens > overlap_tokens:
                break
            result.insert(0, line)
            current_tokens += line_tokens

        return result


# Singleton instance
_default_chunker: Optional[Chunker] = None


def get_chunker(tokens: int = Chunker.DEFAULT_TOKENS, overlap: int = Chunker.DEFAULT_OVERLAP) -> Chunker:
    """Get or create the default chunker instance.

    Args:
        tokens: Target tokens per chunk.
        overlap: Overlap tokens between chunks.

    Returns:
        Chunker instance.
    """
    global _default_chunker

    if _default_chunker is None or \
       _default_chunker.tokens != tokens or \
       _default_chunker.overlap != overlap:
        _default_chunker = Chunker(tokens=tokens, overlap=overlap)

    return _default_chunker
