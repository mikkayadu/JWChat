"""
Scripture lookup
────────────────
Fetches Bible verse text from the public jw.org Bible pages so the UI can
display a verse inline (the browser can't fetch jw.org directly — CORS).

Every language goes through jw.org's finder, which resolves a book number to
that language's chapter page (``/finder?wtlocale=TW&pub=nwt&bible=43003016``
redirects to ``/tw/nhomakorabea/bible/nwt/nhoma/yohane/3/``), so no per-language
book slugs are needed. English uses the study Bible (``nwtsty``).

The chapter pages are server-rendered, so a plain HTTP GET + BeautifulSoup is
enough; no headless browser is required. Each verse lives in a
``<span class="verse" id="v{book}{CCC}{VVV}">`` element; the book number is
not zero-padded (Genesis 1:1 is ``v1001001``, John 3:16 is ``v43003016``).
"""

from __future__ import annotations

import re
from functools import lru_cache
from urllib.parse import urldefrag

import httpx
from loguru import logger

import languages

_FINDER = "https://www.jw.org/finder"

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


def verse_anchor(book_number: int, chapter: int, verse: int) -> str:
    return f"v{book_number}{chapter:03d}{verse:03d}"


def finder_url(lang: languages.Language, book_number: int, chapter: int, verse: int = 1) -> str:
    bible = verse_anchor(book_number, chapter, verse)[1:]
    return f"{_FINDER}?wtlocale={lang.jw_code}&pub={lang.bible_pub}&bible={bible}"


@lru_cache(maxsize=256)
def _fetch_chapter(lang_code: str, book_number: int, chapter: int) -> tuple[str, str]:
    """Fetch a chapter page through the finder; returns (final URL, html).
    Cached — a chapter is requested repeatedly as a user clicks different
    verses in the same passage."""
    lang = languages.get(lang_code)
    resp = httpx.get(
        finder_url(lang, book_number, chapter),
        headers=_HEADERS,
        timeout=20,
        follow_redirects=True,
    )
    resp.raise_for_status()
    url, _ = urldefrag(str(resp.url))
    # Decode from raw bytes so the page's own UTF-8 (curly quotes, NBSP, ɛ/ɔ) survives.
    return url, resp.content.decode("utf-8", errors="replace")


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
    lang_code: str,
    book_number: int,
    chapter: int,
    start_verse: int,
    end_verse: int | None = None,
) -> tuple[str, list[dict]]:
    """Return ``(chapter URL, [{"verse": int, "text": str}, ...])`` for the
    requested range in ``lang_code``'s Bible.

    Raises ScriptureNotFound if none of the requested verses are present.
    """
    from bs4 import BeautifulSoup

    last = end_verse or start_verse
    if last < start_verse:
        last = start_verse

    url, html = _fetch_chapter(lang_code, book_number, chapter)
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
        logger.warning(f"No verse text found for {lang_code} {book_number} {chapter}:{start_verse} ({anchor})")
        raise ScriptureNotFound(f"Verse not found: book {book_number} {chapter}:{start_verse}")

    return url, verses
