"""
HTML Ingestion
─────────────
Scrapes public web pages using Playwright (handles JS-rendered pages)
with a BeautifulSoup fallback for simple static pages.

Strips navigation, footers, ads and other boilerplate automatically.
"""

import os
from pathlib import Path
from urllib.parse import urlparse
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
    "cookies and similar technologies",
    "privacy settings",
]

_MIN_TEXT_CHARS = 180
_MIN_WORD_COUNT = 30

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
_DEFAULT_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
_DEFAULT_ACCEPT_LANGUAGE = "en-US,en;q=0.9"

_CONTENT_SELECTORS = [
    "article#article",
    "article.jwac",
    "main",
    "article",
    "[itemprop='articleBody']",
    "[role='main']",
]


def scrape_url(
    url: str,
    use_playwright: bool = True,
    language: str | None = None,
) -> ExtractedPage | None:
    """
    Scrape a single URL and return its cleaned text content.
    Set use_playwright=False for simple static pages (faster).
    ``language`` overrides the language inferred from the URL.
    """
    fetched = _fetch_html(url, use_playwright)
    if not fetched:
        return None
    raw_html, status_code = fetched
    return _parse_html(raw_html, url, status_code=status_code, language=language)


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
            context = _create_browser_context(browser)
            _apply_site_cookies(context, url)
            page = context.new_page()

            # Block images, fonts, media — speeds up scraping
            page.route("**/*.{png,jpg,jpeg,gif,webp,svg,woff,woff2,ttf,mp4,mp3}", lambda r: r.abort())

            response = page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            _handle_consent_dialog(page)
            _wait_for_content(page)
            try:
                page.wait_for_load_state("networkidle", timeout=5_000)
            except Exception:
                pass
            html = page.content()
            _maybe_dump_html(url, html, stage="after-consent")
            status_code = response.status if response else None
            context.close()
            browser.close()
            logger.debug(f"Playwright fetched: {url}")
            if _is_blocked_response(html=html, status_code=status_code):
                logger.warning(f"Playwright blocked for {url} (status={status_code}) — trying httpx")
                return _fetch_with_httpx(url)
            return html, status_code
    except Exception as e:
        logger.warning(f"Playwright failed for {url}: {e} — trying httpx")
        return _fetch_with_httpx(url)


def _fetch_with_httpx(url: str) -> tuple[str, int | None] | None:
    """Lightweight fallback for static pages."""
    try:
        import httpx
        headers = _build_default_headers()
        resp = httpx.get(url, timeout=15, follow_redirects=True,
                         headers=headers)
        resp.raise_for_status()
        # Handle encoding issues gracefully
        return _decode_response_content(resp), resp.status_code
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None


def _decode_response_content(resp) -> str:
    """Decode response content handling various encodings."""
    content = resp.content
    # Try encoding from response headers first
    encoding = resp.encoding or "utf-8"
    try:
        return content.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        pass
    # Fallback: try common encodings
    for enc in ["utf-8", "latin-1", "windows-1252", "iso-8859-1"]:
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    # Last resort: decode with errors ignored
    return content.decode("utf-8", errors="replace")


def _parse_html(
    html: str,
    url: str,
    status_code: int | None = None,
    language: str | None = None,
) -> ExtractedPage | None:
    """Clean and extract meaningful text from raw HTML."""
    from bs4 import BeautifulSoup
    import languages

    soup = BeautifulSoup(html, "lxml")

    # Remove noise tags entirely
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    # Remove elements whose class/id suggests boilerplate
    # Iterate over a snapshot because decomposing nodes mutates the tree.
    for tag in list(soup.find_all(True)):
        if getattr(tag, "attrs", None) is None:
            continue

        if _is_primary_content_tag(tag):
            continue

        classes = tag.get("class") or []
        if isinstance(classes, str):
            classes = [classes]
        class_id = " ".join(classes) + " " + (tag.get("id") or "")
        if _NOISE_PATTERNS.search(class_id):
            tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else url

    # Prefer article content if present, otherwise fall back to main/body
    content_tag = (
        soup.select_one("article#article")
        or soup.select_one("article.jwac")
        or soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

    # Extract text while preserving heading hierarchy as Markdown
    text = _extract_structured_text(content_tag)

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
            "language": languages.resolve_ingest_language(url, language).code,
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

    if len(normalized_text) < _MIN_TEXT_CHARS and word_count < _MIN_WORD_COUNT:
        return True

    return False


def _is_primary_content_tag(tag) -> bool:
    if tag.name in {"article", "main"}:
        return True

    if tag.get("id") in {"article", "content"}:
        return True

    role = tag.get("role")
    if role == "main":
        return True

    return False


def _build_default_headers() -> dict[str, str]:
    user_agent = os.getenv("SCRAPER_USER_AGENT", _DEFAULT_USER_AGENT)
    accept = os.getenv("SCRAPER_ACCEPT", _DEFAULT_ACCEPT)
    accept_language = os.getenv("SCRAPER_ACCEPT_LANGUAGE", _DEFAULT_ACCEPT_LANGUAGE)
    return {
        "User-Agent": user_agent,
        "Accept": accept,
        "Accept-Language": accept_language,
    }


def _create_browser_context(browser):
    headers = _build_default_headers()
    user_agent = headers.get("User-Agent")
    locale = os.getenv("SCRAPER_LOCALE", "en-US")
    proxy = os.getenv("SCRAPER_PROXY")

    context_kwargs = {
        "user_agent": user_agent,
        "locale": locale,
        "extra_http_headers": headers,
    }
    if proxy:
        context_kwargs["proxy"] = {"server": proxy}

    return browser.new_context(**context_kwargs)


def _is_blocked_response(html: str, status_code: int | None) -> bool:
    if status_code is not None and status_code >= 400:
        return True

    normalized = (html or "").lower()
    sample = normalized[:8000]
    return any(marker in sample for marker in _ERROR_TEXT_MARKERS)


def _handle_consent_dialog(page) -> None:
    """Best-effort cookie/consent handling for JS-rendered sites."""
    import re as _re

    try:
        consent_button = page.locator("button.lnc-acceptCookiesButton")
        if consent_button.count() > 0:
            consent_button.first.click(timeout=2_000)
            return
    except Exception:
        pass

    button_patterns = [
        r"accept all",
        r"accept cookies",
        r"accept",
        r"i accept",
        r"agree",
        r"allow all",
        r"ok",
    ]

    for pattern in button_patterns:
        if _click_consent_button(page, pattern, _re):
            return

    for frame in page.frames:
        for pattern in button_patterns:
            if _click_consent_button(frame, pattern, _re):
                return


def _click_consent_button(target, pattern: str, re_module) -> bool:
    try:
        locator = target.get_by_role("button", name=re_module.compile(pattern, re_module.IGNORECASE))
        if locator.count() > 0:
            locator.first.click(timeout=2_000)
            return True
    except Exception:
        return False
    return False


def _wait_for_content(page) -> None:
    for selector in _CONTENT_SELECTORS:
        try:
            page.wait_for_selector(selector, timeout=5_000)
            return
        except Exception:
            continue


def _maybe_dump_html(url: str, html: str, stage: str) -> None:
    if os.getenv("SCRAPER_DEBUG_DUMP", "").lower() not in {"1", "true", "yes"}:
        return

    safe_name = re.sub(r"[^a-zA-Z0-9]+", "-", url).strip("-")[:120]
    dump_dir = Path(".tmp") / "scraper-dumps"
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = dump_dir / f"{safe_name}-{stage}.html"
    dump_path.write_text(html or "", encoding="utf-8")
    logger.info(f"[HTML] Debug dump saved: {dump_path}")


def _extract_structured_text(tag) -> str:
    """
    Extract text from a BeautifulSoup tag while converting HTML heading tags
    (h1-h4) into Markdown markers (#, ##, ###, ####).

    This preserves the document's section hierarchy so the chunker can use
    headings as natural split boundaries, and the context enricher can label
    each chunk with its nearest section heading.

    Replaces headings directly on the tag (safe — the tag is discarded after
    this call; it's never reused elsewhere in the scraper).
    """
    _HEADING_MAP = {"h1": "#", "h2": "##", "h3": "###", "h4": "####"}

    for level, marker in _HEADING_MAP.items():
        for heading in tag.find_all(level):
            heading_text = heading.get_text(strip=True)
            if heading_text:
                heading.replace_with(f"\n\n{marker} {heading_text}\n\n")

    lines = [ln.strip() for ln in tag.get_text(separator="\n").splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _apply_site_cookies(context, url: str) -> None:
    """Pre-seed consent cookies for known sites to reduce overlays."""
    try:
        hostname = urlparse(url).hostname or ""
        if hostname.endswith("jw.org"):
            context.add_cookies([
                {
                    "name": "cookieConsent-FUNCTIONAL",
                    "value": "true",
                    "domain": ".jw.org",
                    "path": "/",
                },
            ])
    except Exception:
        return
