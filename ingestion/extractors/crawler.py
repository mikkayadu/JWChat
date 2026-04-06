"""
Site Crawler
────────────
Automatically discovers all pages on your website starting from a
seed URL. Follows internal links recursively so you never have to
manually list pages.

Two modes:
  1. Sitemap crawl  — reads /sitemap.xml (fast, no browsing needed)
  2. Full crawl     — follows every <a href> it finds (thorough)

Both modes stay within your domain and respect robots.txt.

Usage:
    from ingestion.crawler import crawl_site, crawl_sitemap

    # Option A: crawl from sitemap (recommended if your site has one)
    urls = crawl_sitemap("https://yourwebsite.com")

    # Option B: full crawl from homepage
    urls = crawl_site("https://yourwebsite.com", max_pages=500)

    # Split by type for ingestion
    pdf_urls   = [u for u in urls if u.endswith(".pdf")]
    video_urls = [u for u in urls if u.endswith((".mp4", ".mp3", ".webm"))]
    page_urls  = [u for u in urls if not u.endswith((".pdf", ".mp4", ".mp3", ".webm"))]
"""

import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag
from loguru import logger


# File extensions we want to capture (not browse into)
MEDIA_EXTENSIONS = {".pdf", ".mp4", ".mp3", ".wav", ".m4a", ".webm", ".ogg"}

# Extensions to skip entirely (images, fonts, scripts, etc.)
SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".css", ".js", ".woff", ".woff2", ".ttf", ".eot",
    ".zip", ".tar", ".gz", ".exe", ".dmg",
}


# ── Public API ────────────────────────────────────────────────────────────────

def crawl_site(
    start_url: str,
    max_pages: int = 1000,
    delay: float = 0.5,
    respect_robots: bool = True,
    include_subdomains: bool = False,
) -> list[str]:
    """
    Crawl a website starting from start_url, following all internal links.

    Returns a list of discovered URLs (HTML pages + PDFs + videos).

    Args:
        start_url:          The homepage or starting page to crawl from
        max_pages:          Safety cap — stop after this many pages
        delay:              Seconds to wait between requests (be polite)
        respect_robots:     Whether to honour robots.txt disallow rules
        include_subdomains: Whether to follow links to subdomains
    """
    parsed = urlparse(start_url)
    base_domain = parsed.netloc
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    disallowed = _load_robots(base_url) if respect_robots else set()

    visited: set[str] = set()
    queue: deque[str] = deque([_normalise(start_url)])
    found: list[str] = []

    logger.info(f"Starting crawl from: {start_url}")
    logger.info(f"Domain: {base_domain} | Max pages: {max_pages} | Delay: {delay}s")

    while queue and len(found) < max_pages:
        url = queue.popleft()

        if url in visited:
            continue
        visited.add(url)

        # Skip if robots.txt disallows it
        if _is_disallowed(url, disallowed):
            logger.debug(f"  Skipped (robots.txt): {url}")
            continue

        # Media files — add to found list but don't crawl into them
        ext = _get_extension(url)
        if ext in MEDIA_EXTENSIONS:
            found.append(url)
            logger.info(f"  Found media: {url}")
            continue

        if ext in SKIP_EXTENSIONS:
            continue

        # Fetch and parse the page
        html = _fetch(url)
        if not html:
            continue

        found.append(url)
        logger.info(f"  [{len(found)}/{max_pages}] {url}")

        # Extract and queue all internal links
        links = _extract_links(html, url, base_domain, include_subdomains)
        for link in links:
            if link not in visited:
                queue.append(link)

        if delay > 0:
            time.sleep(delay)

    logger.success(f"Crawl complete: {len(found)} URLs discovered")
    return found


def crawl_sitemap(base_url: str) -> list[str]:
    """
    Read your site's sitemap.xml and return all URLs in it.
    Much faster than a full crawl — use this if your site has a sitemap.

    Tries common sitemap locations:
      /sitemap.xml
      /sitemap_index.xml
      /sitemap/sitemap.xml

    Also handles sitemap index files (sitemaps that link to other sitemaps).
    """
    base_url = base_url.rstrip("/")
    sitemap_candidates = [
        f"{base_url}/sitemap.xml",
        f"{base_url}/sitemap_index.xml",
        f"{base_url}/sitemap/sitemap.xml",
        f"{base_url}/page-sitemap.xml",
    ]

    # Also check robots.txt for a Sitemap: directive
    robots_sitemap = _get_sitemap_from_robots(base_url)
    if robots_sitemap:
        sitemap_candidates.insert(0, robots_sitemap)

    for sitemap_url in sitemap_candidates:
        urls = _parse_sitemap(sitemap_url)
        if urls:
            logger.success(f"Sitemap found at {sitemap_url}: {len(urls)} URLs")
            return urls

    logger.warning("No sitemap found. Use crawl_site() for a full crawl instead.")
    return []


def save_url_lists(urls: list[str], output_dir: str = "."):
    """
    Split discovered URLs into separate files ready for pipeline.py:
      page_urls.txt   — HTML pages
      pdf_links.txt   — PDF files
      video_links.txt — video/audio files
    """
    from pathlib import Path

    out = Path(output_dir)
    pages  = [u for u in urls if _get_extension(u) not in MEDIA_EXTENSIONS]
    pdfs   = [u for u in urls if _get_extension(u) == ".pdf"]
    videos = [u for u in urls if _get_extension(u) in {".mp4", ".mp3", ".wav", ".m4a", ".webm", ".ogg"}]

    def write(name, items):
        path = out / name
        path.write_text("\n".join(items))
        logger.info(f"Wrote {len(items)} URLs to {path}")

    write("page_urls.txt",   pages)
    write("pdf_links.txt",   pdfs)
    write("video_links.txt", videos)

    return {"pages": len(pages), "pdfs": len(pdfs), "videos": len(videos)}


# ── Internals ─────────────────────────────────────────────────────────────────

def _fetch(url: str) -> str | None:
    try:
        import httpx
        resp = httpx.get(
            url, timeout=15, follow_redirects=True,
            headers={"User-Agent": "RAG-Crawler/1.0 (respectful bot)"}
        )
        resp.raise_for_status()
        # Only parse HTML responses
        ct = resp.headers.get("content-type", "")
        if "html" not in ct and "xml" not in ct:
            return None
        return resp.text
    except Exception as e:
        logger.debug(f"  Fetch failed: {url}: {e}")
        return None


def _extract_links(
    html: str,
    current_url: str,
    base_domain: str,
    include_subdomains: bool,
) -> list[str]:
    """Extract all internal links from an HTML page."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    links = []

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
            continue

        full_url = _normalise(urljoin(current_url, href))
        parsed = urlparse(full_url)

        # Stay on same domain (optionally allow subdomains)
        if include_subdomains:
            if not parsed.netloc.endswith(base_domain):
                continue
        else:
            if parsed.netloc != base_domain:
                continue

        links.append(full_url)

    return links


def _parse_sitemap(url: str) -> list[str]:
    """Parse a sitemap XML and return all <loc> URLs. Handles sitemap indexes."""
    xml = _fetch(url)
    if not xml:
        return []

    # Extract all <loc> values
    locs = re.findall(r"<loc>\s*(.*?)\s*</loc>", xml)

    # If this is a sitemap index (contains other sitemaps), recurse
    if "<sitemapindex" in xml:
        all_urls = []
        for sitemap_url in locs:
            all_urls.extend(_parse_sitemap(sitemap_url))
        return all_urls

    return locs


def _get_sitemap_from_robots(base_url: str) -> str | None:
    """Check robots.txt for a Sitemap: directive."""
    robots_text = _fetch(f"{base_url}/robots.txt")
    if not robots_text:
        return None
    match = re.search(r"^Sitemap:\s*(.+)$", robots_text, re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip() if match else None


def _load_robots(base_url: str) -> set[str]:
    """Parse robots.txt and return disallowed paths for all bots."""
    robots_text = _fetch(f"{base_url}/robots.txt")
    if not robots_text:
        return set()

    disallowed = set()
    in_relevant_section = False

    for line in robots_text.splitlines():
        line = line.strip()
        if line.lower().startswith("user-agent:"):
            agent = line.split(":", 1)[1].strip()
            in_relevant_section = agent in ("*", "RAG-Crawler")
        elif in_relevant_section and line.lower().startswith("disallow:"):
            path = line.split(":", 1)[1].strip()
            if path:
                disallowed.add(path)

    return disallowed


def _is_disallowed(url: str, disallowed: set[str]) -> bool:
    path = urlparse(url).path
    return any(path.startswith(d) for d in disallowed)


def _normalise(url: str) -> str:
    """Remove fragments and trailing slashes for consistent deduplication."""
    url, _ = urldefrag(url)
    return url.rstrip("/") if url.count("/") > 2 else url


def _get_extension(url: str) -> str:
    path = urlparse(url).path.lower()
    dot = path.rfind(".")
    return path[dot:] if dot != -1 else ""
