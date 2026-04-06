"""
HTML Ingestion
──────────────
Scrapes public web pages using Playwright (handles JS-rendered pages)
with a BeautifulSoup fallback for simple static pages.

Strips navigation, footers, ads and other boilerplate automatically.
"""

import re
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ExtractedPage:
    url: str
    title: str
    text: str
    source_path: str = ""
    metadata: dict = field(default_factory=dict)


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


def scrape_url(url: str, use_playwright: bool = True) -> ExtractedPage | None:
    """
    Scrape a single URL and return its cleaned text content.
    Set use_playwright=False for simple static pages (faster).
    """
    raw_html = _fetch_html(url, use_playwright)
    if not raw_html:
        return None
    return _parse_html(raw_html, url)


def scrape_urls(urls: list[str], use_playwright: bool = True) -> list[ExtractedPage]:
    """Scrape multiple URLs. Returns only successful results."""
    results = []
    for url in urls:
        page = scrape_url(url, use_playwright)
        if page and page.text:
            results.append(page)
    return results


def _fetch_html(url: str, use_playwright: bool) -> str | None:
    if use_playwright:
        return _fetch_with_playwright(url)
    return _fetch_with_httpx(url)


def _fetch_with_playwright(url: str) -> str | None:
    """Use headless browser — handles JS-rendered pages."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Block images, fonts, media — speeds up scraping
            page.route("**/*.{png,jpg,jpeg,gif,webp,svg,woff,woff2,ttf,mp4,mp3}", lambda r: r.abort())

            page.goto(url, wait_until="networkidle", timeout=30_000)
            html = page.content()
            browser.close()
            logger.debug(f"Playwright fetched: {url}")
            return html
    except Exception as e:
        logger.warning(f"Playwright failed for {url}: {e} — trying httpx")
        return _fetch_with_httpx(url)


def _fetch_with_httpx(url: str) -> str | None:
    """Lightweight fallback for static pages."""
    try:
        import httpx
        resp = httpx.get(url, timeout=15, follow_redirects=True,
                         headers={"User-Agent": "RAG-Bot/1.0 (+https://yoursite.com/bot)"})
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None


def _parse_html(html: str, url: str) -> ExtractedPage:
    """Clean and extract meaningful text from raw HTML."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")

    # Remove noise tags entirely
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    # Remove elements whose class/id suggests boilerplate
    for tag in soup.find_all(True):
        class_id = " ".join(tag.get("class", [])) + " " + (tag.get("id") or "")
        if _NOISE_PATTERNS.search(class_id):
            tag.decompose()

    title = soup.title.string.strip() if soup.title else url

    # Prefer <main> or <article> if present, otherwise whole body
    content_tag = soup.find("main") or soup.find("article") or soup.body or soup

    # Get text and clean whitespace
    lines = [line.strip() for line in content_tag.get_text(separator="\n").splitlines()]
    text = "\n".join(line for line in lines if line)

    logger.debug(f"Parsed {url}: {len(text)} chars, title='{title[:60]}'")

    return ExtractedPage(
        url=url,
        title=title,
        text=text,
        metadata={
            "file_type": "html",
            "source_url": url,
            "title": title,
        }
    )
