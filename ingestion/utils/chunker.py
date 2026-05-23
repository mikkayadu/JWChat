"""
Text Chunker
───────────
Splits long text into overlapping chunks suitable for embedding.

Supports multiple strategies:
  1. Recursive - Respects document structure (paragraphs, sentences, words)
  2. Semantic - Splits at topic boundaries using embeddings
  3. Fixed - Original sliding window approach (backward compatible)

Chunking Strategy Selection:
  - Use 'recursive' for most documents (respects structure)
  - Use 'semantic' for narrative content with topic shifts
  - Use 'fixed' for simple text or backward compatibility
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
from loguru import logger


class ChunkingStrategy(str, Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    FIXED = "fixed"


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    token_count: int
    source: str          # file path or URL
    metadata: dict = field(default_factory=dict)
    strategy: str = "fixed"  # Track which strategy created this chunk


def chunk_text(
    text: str,
    source: str,
    metadata: dict | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 100,
    model: str = "text-embedding-3-small",
    strategy: ChunkingStrategy | str = ChunkingStrategy.RECURSIVE,
    separators: list[str] | None = None,
) -> list[TextChunk]:
    """
    Split text into overlapping chunks using the specified strategy.

    Args:
        text:          The raw text to chunk
        source:        Where this text came from (URL or file path)
        metadata:      Extra fields to attach to every chunk
        chunk_size:    Max tokens per chunk (default: 512)
        chunk_overlap: How many tokens to repeat between consecutive chunks (default: 100)
        model:         Tokeniser model name (determines token counts)
        strategy:      Chunking strategy: 'recursive', 'semantic', or 'fixed'
        separators:     Custom separators for recursive chunking (optional)

    Returns:
        List of TextChunk objects ready for embedding
    """
    if not text or not text.strip():
        return []

    strategy = ChunkingStrategy(strategy) if isinstance(strategy, str) else strategy

    # Get token encoder for accurate token counts (used by fixed strategy and metadata)
    enc = _get_encoder(model)

    if strategy == ChunkingStrategy.RECURSIVE:
        raw_chunks = _recursive_chunk(text, chunk_size, chunk_overlap, separators)
    elif strategy == ChunkingStrategy.SEMANTIC:
        raw_chunks = _semantic_chunk(text, chunk_size, chunk_overlap)
    else:  # FIXED
        paragraphs = _split_paragraphs(text)
        raw_chunks = _sliding_window(paragraphs, enc, chunk_size, chunk_overlap)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        token_count = len(enc.encode(chunk_text))
        chunks.append(TextChunk(
            text=chunk_text,
            chunk_index=i,
            token_count=token_count,
            source=source,
            metadata={
                **(metadata or {}),
                "chunk_index": i,
                "total_chunks": len(raw_chunks),  # will update below
                "token_count": token_count,
                "source": source,
                "chunking_strategy": strategy.value,
            },
            strategy=strategy.value,
        ))

    # Now we know the real total, update it
    for c in chunks:
        c.metadata["total_chunks"] = len(chunks)

    logger.debug(f"Chunked '{source}' using {strategy.value}: {len(chunks)} chunks from {len(text)} chars")
    return chunks


def chunk_pages(
    pages: list,
    chunk_size: int = 512,
    chunk_overlap: int = 100,
    strategy: ChunkingStrategy | str = ChunkingStrategy.RECURSIVE,
) -> list[TextChunk]:
    """
    Convenience wrapper: chunk a list of ExtractedPage / TranscribedVideo objects.
    Each page becomes its own set of chunks (page boundaries are respected).

    Args:
        pages:          List of page objects (PDF, HTML, Video)
        chunk_size:    Max tokens per chunk
        chunk_overlap:  Overlap between chunks
        strategy:      Chunking strategy to use
    """
    all_chunks: list[TextChunk] = []

    for page in pages:
        # Support both PDF pages and HTML/video objects
        text = getattr(page, "text", None) or getattr(page, "transcript", "")
        source = getattr(page, "source_path", None) or getattr(page, "url", None) or getattr(page, "source", "unknown")
        metadata = getattr(page, "metadata", {}).copy()

        # Add page number if available
        if hasattr(page, "page_number"):
            metadata["page_number"] = page.page_number

        chunks = chunk_text(
            text,
            source=source,
            metadata=metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
        )
        all_chunks.extend(chunks)

    return all_chunks


# ── Internals ────────────────────────────────────────────────────────────────

def _get_encoder(model: str):
    try:
        import tiktoken
        # tiktoken doesn't know embedding models by name — map to a known one
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")  # works for all OpenAI models
    except ImportError:
        logger.warning("tiktoken not installed — falling back to word-count approximation")
        return _FakeEncoder()


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; further split very long single-line paragraphs on sentences."""
    raw = re.split(r"\n{2,}", text)
    paragraphs = []

    for para in raw:
        para = para.strip()
        if not para:
            continue
        # Split run-on paragraphs on sentence boundaries
        if len(para) > 1500:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            paragraphs.extend(s.strip() for s in sentences if s.strip())
        else:
            paragraphs.append(para)

    return paragraphs


def _sliding_window(
    paragraphs: list[str],
    enc,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Greedy bin-packing with overlap.
    Fills each chunk up to chunk_size tokens, then starts a new chunk
    beginning chunk_overlap tokens back into the previous one.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(enc.encode(para))

        # Single paragraph is larger than chunk_size — hard-split it
        if para_tokens > chunk_size:
            if current:
                chunks.append("\n\n".join(current))
                current, current_tokens = [], 0
            sub_chunks = _hard_split(para, enc, chunk_size, chunk_overlap)
            chunks.extend(sub_chunks)
            continue

        if current_tokens + para_tokens > chunk_size:
            # Save current chunk
            chunks.append("\n\n".join(current))
            # Seed next chunk with overlap (work backwards from end)
            overlap_paras, overlap_tokens = [], 0
            for p in reversed(current):
                pt = len(enc.encode(p))
                if overlap_tokens + pt > chunk_overlap:
                    break
                overlap_paras.insert(0, p)
                overlap_tokens += pt
            current = overlap_paras
            current_tokens = overlap_tokens

        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return [c for c in chunks if c.strip()]


def _hard_split(text: str, enc, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Token-level split for paragraphs that exceed chunk_size on their own."""
    tokens = enc.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start += chunk_size - chunk_overlap

    return chunks


def _recursive_chunk(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 100,
    separators: list[str] | None = None,
) -> list[str]:
    """
    Use LangChain's RecursiveCharacterTextSplitter for structure-aware chunking.
    Respects document structure (paragraphs > sentences > words).

    Uses tiktoken to measure chunk size in tokens (not characters) so that
    chunk_size matches the embedding model's token budget accurately.
    """
    try:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        splitter = RecursiveCharacterTextSplitter(
            separators=separators or ["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda text: len(enc.encode(text)),
        )
        return splitter.split_text(text)
    except ImportError:
        logger.warning("LangChain or tiktoken not installed. Falling back to fixed chunking.")
        return _fallback_chunk(text, chunk_size, chunk_overlap)


def _semantic_chunk(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 100,
) -> list[str]:
    """
    Semantic chunking: split at topic boundaries using embedding similarity.
    More expensive but produces semantically coherent chunks.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain.embeddings import OpenAIEmbeddings

        # Use OpenAI embeddings for semantic splitting
        embeddings = OpenAIEmbeddings()
        splitter = SemanticChunker(
            embeddings=embeddings,
            buffer_size=1,  # Number of sentences to group
        )
        return splitter.split_text(text)
    except ImportError:
        logger.warning("langchain_experimental not installed. Falling back to recursive chunking.")
        return _recursive_chunk(text, chunk_size, chunk_overlap)


def _fallback_chunk(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Simple character-based chunking as fallback."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


class _FakeEncoder:
    """Approximate token counter when tiktoken isn't available (1 token ≈ 4 chars)."""
    def encode(self, text: str) -> list:
        return list(range(max(1, len(text) // 4)))

    def decode(self, tokens: list) -> str:
        # Can't actually decode without tiktoken — return placeholder
        return "[chunk]"
