"""
HTML Ingestion
─────────────
Scrapes public web pages using Playwright (handles JS-rendered pages)
with a BeautifulSoup fallback for simple static pages.

Strips navigation, footers, ads and other boilerplate automatically.
"""

import re
from loguru import logger

from ingestion.extractors import ExtractedPage


# HTML tags whose content we always discard
_NOISE_TAGS = [
    "nav", "footer", "header", "aside", "script",
    "style", "noscript", "iframe", "form", "button",
    "svg", "img", "figure", "figcaption", "advertisement",
]

# CSS classes/IDs that usually signal boilerplate
_NOISE_PATTERNS = re.compile(
    r"(nav|menu|sidebar|footer|header|cookie|banner|ad-|advertisement|social|share)",
    re.IGNORECASE
)

_ERROR_TITLE_PATTERNS = re.compile(
    r"(request could not be satisfied|access denied|forbidden|not found|captcha|blocked)",
    re.IGNORECASE,
)

_ERROR_TEXT_MARKERS = [
    "error: the request could not be satisfied",
    "cloudfront",
    "access denied",
    "forbidden",
    "not found",
    "request blocked",
    "verify you are human",
    "captcha",
    "temporarily unavailable",
]

_MIN_TEXT_CHARS = 180
_MIN_WORD_COUNT = 30


def scrape_url(url: str, use_playwright: bool = True) -> ExtractedPage | None:
    """
    Scrape a single URL and return its cleaned text content.
    Set use_playwright=False for simple static pages (faster).
    """
    fetched = _fetch_html(url, use_playwright)
    if not fetched:
        return None
    raw_html, status_code = fetched
    return _parse_html(raw_html, url, status_code=status_code)


def scrape_urls(urls: list[str], use_playwright: bool = True) -> list[ExtractedPage]:
    """Scrape multiple URLs. Returns only successful results."""
    results = []
    for url in urls:
        page = scrape_url(url, use_playwright)
        if page and page.text:
            results.append(page)
    return results


def _fetch_html(url: str, use_playwright: bool) -> tuple[str, int | None] | None:
    if use_playwright:
        return _fetch_with_playwright(url)
    return _fetch_with_httpx(url)


def _fetch_with_playwright(url: str) -> tuple[str, int | None] | None:
    """Use headless browser — handles JS-rendered pages."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Block images, fonts, media — speeds up scraping
            page.route("**/*.{png,jpg,jpeg,gif,webp,svg,woff,woff2,ttf,mp4,mp3}", lambda r: r.abort())

            response = page.goto(url, wait_until="networkidle", timeout=30_000)
            html = page.content()
            status_code = response.status if response else None
            browser.close()
            logger.debug(f"Playwright fetched: {url}")
            return html, status_code
    except Exception as e:
        logger.warning(f"Playwright failed for {url}: {e} — trying httpx")
        return _fetch_with_httpx(url)


def _fetch_with_httpx(url: str) -> tuple[str, int | None] | None:
    """Lightweight fallback for static pages."""
    try:
        import httpx
        resp = httpx.get(url, timeout=15, follow_redirects=True,
                         headers={"User-Agent": "RAG-Bot/1.0 (+https://yoursite.com/bot)"})
        resp.raise_for_status()
        return resp.text, resp.status_code
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None


def _parse_html(html: str, url: str, status_code: int | None = None) -> ExtractedPage | None:
    """Clean and extract meaningful text from raw HTML."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")

    # Remove noise tags entirely
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    # Remove elements whose class/id suggests boilerplate
    # Iterate over a snapshot because decomposing nodes mutates the tree.
    for tag in list(soup.find_all(True)):
        if getattr(tag, "attrs", None) is None:
            continue

        classes = tag.get("class") or []
        if isinstance(classes, str):
            classes = [classes]
        class_id = " ".join(classes) + " " + (tag.get("id") or "")
        if _NOISE_PATTERNS.search(class_id):
            tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else url

    # Prefer <main> or <article> if present, otherwise whole body
    content_tag = soup.find("main") or soup.find("article") or soup.body or soup

    # Get text and clean whitespace
    lines = [line.strip() for line in content_tag.get_text(separator="\n").splitlines()]
    text = "\n".join(line for line in lines if line)

    if _should_skip_page(title=title, text=text, status_code=status_code):
        logger.warning(
            f"Skipping non-content page: {url} | status={status_code} | title='{title[:80]}'"
        )
        return None

    logger.debug(f"Parsed {url}: {len(text)} chars, title='{title[:60]}'")

    return ExtractedPage(
        url=url,
        title=title,
        text=text,
        metadata={
            "file_type": "html",
            "source_url": url,
            "title": title,
            "http_status": status_code,
        }
    )


def _should_skip_page(title: str, text: str, status_code: int | None) -> bool:
    if status_code is not None and status_code >= 400:
        return True

    title_text = (title or "").strip()
    normalized_text = (text or "").strip().lower()

    sample = normalized_text[:6000]
    if any(marker in sample for marker in _ERROR_TEXT_MARKERS):
        return True

    word_count = len(re.findall(r"\b\w+\b", normalized_text))
    if _ERROR_TITLE_PATTERNS.search(title_text) and word_count < 180:
        return True

    if len(normalized_text) < _MIN_TEXT_CHARS or word_count < _MIN_WORD_COUNT:
        return True

    return False
