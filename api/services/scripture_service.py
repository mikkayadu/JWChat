"""
Scripture lookup
────────────────
Fetches Bible verse text from the public jw.org study-bible pages so the UI can
display a verse inline (the browser can't fetch jw.org directly — CORS).

The study-bible chapter pages are server-rendered, so a plain HTTP GET +
BeautifulSoup is enough; no headless browser is required. Each verse lives in a
``<span class="verse" id="vBBCCCVVV">`` element whose id matches the anchor the
frontend already computes (2-digit book, 3-digit chapter, 3-digit verse).
"""

from __future__ import annotations

import re
from functools import lru_cache

import httpx
from loguru import logger

_BASE = "https://www.jw.org/en/library/bible/study-bible/books"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Elements inside a verse span that are markup, not scripture text.
_STRIP_SELECTORS = [
    ("sup", "verseNum"),     # the verse number
    ("span", "chapterNum"),  # chapter number shown on verse 1
    ("a", "xrefLink"),       # cross-reference markers (rendered as "+")
    ("a", "footnoteLink"),   # footnote markers (rendered as "*")
    ("a", "fn"),             # alternate footnote class
]


class ScriptureNotFound(Exception):
    """Raised when a verse can't be located on the fetched page."""


def chapter_url(slug: str, chapter: int) -> str:
    return f"{_BASE}/{slug}/{chapter}/"


def verse_anchor(book_number: int, chapter: int, verse: int) -> str:
    return f"v{book_number:02d}{chapter:03d}{verse:03d}"


@lru_cache(maxsize=256)
def _fetch_chapter_html(slug: str, chapter: int) -> str:
    """Fetch a chapter page. Cached — a chapter is requested repeatedly as a
    user clicks different verses in the same passage."""
    url = chapter_url(slug, chapter)
    resp = httpx.get(url, headers=_HEADERS, timeout=20, follow_redirects=True)
    resp.raise_for_status()
    # Decode from raw bytes so the page's own UTF-8 (curly quotes, NBSP) survives.
    return resp.content.decode("utf-8", errors="replace")


def _clean_verse_text(span) -> str:
    for name, klass in _STRIP_SELECTORS:
        for el in span.select(f"{name}.{klass}"):
            el.decompose()
    text = span.get_text(" ", strip=True)
    # Collapse whitespace and tidy punctuation spacing left by removed markers.
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def get_verses(
    slug: str,
    chapter: int,
    book_number: int,
    start_verse: int,
    end_verse: int | None = None,
) -> list[dict]:
    """Return ``[{"verse": int, "text": str}, ...]`` for the requested range.

    Raises ScriptureNotFound if none of the requested verses are present.
    """
    from bs4 import BeautifulSoup

    last = end_verse or start_verse
    if last < start_verse:
        last = start_verse

    html = _fetch_chapter_html(slug, chapter)
    soup = BeautifulSoup(html, "lxml")

    verses: list[dict] = []
    for verse in range(start_verse, last + 1):
        anchor = verse_anchor(book_number, chapter, verse)
        span = soup.find("span", class_="verse", id=anchor)
        if span is None:
            continue
        text = _clean_verse_text(span)
        if text:
            verses.append({"verse": verse, "text": text})

    if not verses:
        logger.warning(f"No verse text found for {slug} {chapter}:{start_verse} ({anchor})")
        raise ScriptureNotFound(f"Verse not found: {slug} {chapter}:{start_verse}")

    return verses
