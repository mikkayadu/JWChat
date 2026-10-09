"""
Text Pre-processor
──────────────────
Sits between the scraper and the chunker. Applies four passes:

  1. normalize_text        — Unicode NFKC, curly quotes, em dashes, zero-width
                             chars, citation markers, whitespace collapse
  2. remove_boilerplate    — Drop lines below information density threshold
                             (< 4 words, known noise phrases)
  3. deduplicate_paragraphs — Remove repeated paragraphs by fingerprint
  4. enrich_chunks         — Prepend [Source | Section] context header to every
                             chunk before embedding (Anthropic Contextual Retrieval)

Usage:
    page   = scrape_url(url)
    page   = preprocess(page)
    chunks = chunk_text(page.text, ...)
    chunks = enrich_chunks_with_context(chunks, page.title, page.text)
    embed_chunks(chunks, ...)
"""

import re
import unicodedata
from loguru import logger

from ingestion.extractors import ExtractedPage
from ingestion.utils.chunker import TextChunk


# ── Constants ─────────────────────────────────────────────────────────────────

_CITATION_RE = re.compile(r"\[\d+\]|\(\d+\)|\^\d+")
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")

_BOILERPLATE_RE = re.compile(
    r"^("
    r"share|read\s+more|back\s+to\s+top|subscribe|sign\s+up|click\s+here|"
    r"learn\s+more|see\s+more|view\s+all|load\s+more|follow\s+us|contact\s+us|"
    r"home|search|menu|close|next|previous|print|download|"
    r"you\s+might\s+also\s+like|recommended|advertisement|"
    r"skip\s+to\s+(content|main\s+content)|"
    r"cookie\s+policy|privacy\s+policy|terms\s+of\s+(use|service)|"
    r"all\s+rights\s+reserved|copyright|©|\d+"
    r")$",
    re.IGNORECASE,
)

_MIN_WORDS_PER_LINE = 4

# A chapter:verse pair ("Exodus 20:3", "Yohane 3:16"). A short line holding
# one is a scripture reference, not boilerplate.
_SCRIPTURE_REF_RE = re.compile(r"\d+:\d+")


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess(page: ExtractedPage) -> ExtractedPage:
    """
    Run the full preprocessing pipeline on a scraped page.
    Returns a new ExtractedPage with cleaned text; all other fields unchanged.
    """
    original_len = len(page.text)

    text = normalize_text(page.text)
    text = remove_boilerplate_lines(text)
    text = deduplicate_paragraphs(text)

    reduction = 100 * (1 - len(text) / max(original_len, 1))
    logger.debug(
        f"Preprocessed '{page.url or page.source_path}': "
        f"{original_len} → {len(text)} chars ({reduction:.0f}% reduced)"
    )

    return ExtractedPage(
        text=text,
        source_path=page.source_path,
        metadata=page.metadata,
        url=page.url,
        title=page.title,
        page_number=page.page_number,
        is_ocr=page.is_ocr,
    )


def normalize_text(text: str) -> str:
    """
    Light normalization — preserves case, punctuation, acronyms, and
    proper nouns. Only fixes encoding artifacts and whitespace.

    Deliberately does NOT lowercase or strip punctuation because those
    destroy meaningful signals (acronyms, proper nouns, quoted phrases).
    """
    if not text:
        return ""

    # Unicode NFKC: resolves ligatures, half-width chars, compatibility forms
    text = unicodedata.normalize("NFKC", text)

    # Curly/smart quotes → ASCII equivalents
    text = (
        text.replace("“", '"').replace("”", '"')
            .replace("‘", "'").replace("’", "'")
    )

    # Em / en dashes → spaced hyphen (preserves readability)
    text = text.replace("—", " - ").replace("–", " - ")

    # Zero-width and invisible characters
    text = (
        text.replace("​", "")   # zero-width space
            .replace("‌", "")   # zero-width non-joiner
            .replace("‍", "")   # zero-width joiner
            .replace("﻿", "")   # BOM
            .replace("­", "")   # soft hyphen
    )

    # Strip citation markers: [1], (2), ^3
    text = _CITATION_RE.sub("", text)

    # Collapse inline whitespace per line, keep newlines intact
    lines = [_INLINE_WHITESPACE_RE.sub(" ", ln).strip() for ln in text.splitlines()]
    text = "\n".join(lines)

    # Collapse 3+ consecutive blank lines into exactly 2
    text = _EXCESS_BLANK_LINES_RE.sub("\n\n", text)

    return text.strip()


def remove_boilerplate_lines(text: str) -> str:
    """
    Drop lines that carry no useful semantic content:
      - Blank lines are always kept (they mark paragraph boundaries)
      - Markdown headings (# ...) are always kept
      - Lines matching known boilerplate phrases are dropped
      - Lines with fewer than _MIN_WORDS_PER_LINE words are dropped, unless
        they hold a scripture reference
    """
    filtered = []
    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            filtered.append(line)
            continue

        if stripped.startswith("#"):
            filtered.append(line)
            continue

        if _BOILERPLATE_RE.match(stripped):
            continue

        if (
            len(re.findall(r"\b\w+\b", stripped)) < _MIN_WORDS_PER_LINE
            and not _SCRIPTURE_REF_RE.search(stripped)
        ):
            continue

        filtered.append(line)

    return "\n".join(filtered)


def deduplicate_paragraphs(text: str) -> str:
    """
    Remove repeated paragraphs by exact fingerprint (whitespace-collapsed,
    lowercased). Keeps the first occurrence; drops subsequent duplicates.

    Handles mirror pages, repeated disclaimer blocks, and boilerplate footers
    that slipped through HTML-level filtering.
    """
    paragraphs = re.split(r"\n{2,}", text)
    seen: set[str] = set()
    unique: list[str] = []

    for para in paragraphs:
        fingerprint = re.sub(r"\s+", " ", para.strip().lower())
        if not fingerprint:
            continue
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(para.strip())

    return "\n\n".join(unique)


# ── Contextual enrichment ────────────────────────────────────────────────────

def enrich_chunks_with_context(
    chunks: list[TextChunk],
    title: str,
    full_text: str,
) -> list[TextChunk]:
    """
    Anthropic Contextual Retrieval: prepend a short context header to each
    chunk before it is embedded.

    This is a rule-based implementation (title + nearest Markdown heading)
    which is free and instantaneous — no LLM call needed. The heading already
    carries the semantic label for the section so retrieval quality improves
    significantly without added latency or cost.

    Header format (50-80 chars):
        [Source: {title} | Section: {nearest heading}]

    Reference: https://www.anthropic.com/news/contextual-retrieval
    """
    heading_index = _build_heading_index(full_text)

    enriched: list[TextChunk] = []
    for chunk in chunks:
        heading = _nearest_heading(chunk.text, full_text, heading_index)

        prefix = f"[Source: {title}"
        if heading:
            prefix += f" | Section: {heading}"
        prefix += "]"

        enriched_text = f"{prefix}\n{chunk.text}"

        enriched.append(TextChunk(
            text=enriched_text,
            chunk_index=chunk.chunk_index,
            token_count=chunk.token_count,
            source=chunk.source,
            metadata={**chunk.metadata, "context_prefix": prefix},
            strategy=chunk.strategy,
        ))

    return enriched


# ── Internals ─────────────────────────────────────────────────────────────────

def _build_heading_index(text: str) -> dict[int, str]:
    """Map character offsets → heading text for nearest-heading lookup."""
    index: dict[int, str] = {}
    for match in re.finditer(r"^#{1,4}\s+(.+)$", text, re.MULTILINE):
        index[match.start()] = match.group(1).strip()
    return index


def _nearest_heading(chunk_text: str, full_text: str, heading_index: dict[int, str]) -> str:
    """
    Find the closest heading that appears before this chunk in the full text.
    Uses the first 80 characters of the chunk as a search anchor.
    """
    if not heading_index:
        return ""

    anchor = chunk_text[:80].strip()
    pos = full_text.find(anchor)
    if pos == -1:
        return ""

    best_offset, best_heading = -1, ""
    for offset, heading in heading_index.items():
        if offset <= pos and offset > best_offset:
            best_offset = offset
            best_heading = heading

    return best_heading
