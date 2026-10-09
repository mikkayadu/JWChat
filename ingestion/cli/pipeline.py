"""
Ingestion Pipeline — Entry Point
──────────────────────────────────
Ingest your website content without downloading anything.
Just provide URLs — PDFs and videos are streamed directly.

Usage examples:

  # Provide a file of PDF URLs (one per line)
  python pipeline.py --pdf-urls pdf_links.txt

  # Provide a file of video URLs (one per line)
  python pipeline.py --video-urls video_links.txt

  # Scrape HTML pages
  python pipeline.py --urls page_urls.txt

  # Everything at once
  python pipeline.py --pdf-urls pdf_links.txt --video-urls video_links.txt --urls page_urls.txt

  # Local files still work too
  python pipeline.py --pdfs ./docs --videos ./videos

  # Test without Celery (slower, no queue needed)
  python pipeline.py --pdf-urls pdf_links.txt --sync

  # Crawl one jw.org language; the crawl stays under /tw/ and vectors go to
  # the Twi namespace (the language is read from the seed URL)
  python pipeline.py --crawl https://www.jw.org/tw/nhomakorabea/ --max-pages 300

  # Force a language for non-jw.org sources
  python pipeline.py --pdf-urls twi_pdfs.txt --language tw
"""

import argparse
import sys
import uuid
from pathlib import Path
from urllib.parse import (
    parse_qsl,
    urlencode,
    urldefrag,
    urlparse,
    urlunparse,
)
from loguru import logger
from dotenv import load_dotenv

import languages

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="RAG Ingestion Pipeline")

    # Auto-crawl (no URL lists needed)
    parser.add_argument("--crawl",   help="Auto-crawl entire site from this URL (e.g. https://yoursite.com)")
    parser.add_argument("--sitemap", help="Use sitemap.xml to discover all URLs (faster than --crawl)")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages to crawl (default: no limit)")
    parser.add_argument("--crawl-delay", type=float, default=0.5, help="Seconds between requests (default: 0.5)")

    # URL-based inputs (no downloading needed)
    parser.add_argument("--pdf-urls",   help="Text file with PDF URLs, one per line")
    parser.add_argument("--video-urls", help="Text file with video URLs, one per line")
    parser.add_argument("--urls",       help="Text file with HTML page URLs, one per line")

    # Local file inputs (optional)
    parser.add_argument("--pdfs",   help="Directory of local PDF files")
    parser.add_argument("--videos", help="Directory of local video files")

    parser.add_argument("--sync",   action="store_true",
                        help="Run synchronously without Celery (good for testing)")
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size for video transcription (default: base)")
    parser.add_argument("--language", choices=sorted(languages.LANGUAGES),
                        help="Language to tag every ingested item with. Default: read from "
                             "each jw.org URL's locale, else English")
    return parser.parse_args()


def run_pipeline(args):
    jobs = 0

    if args.crawl:
        jobs += ingest_by_crawl(args.crawl, max_pages=args.max_pages,
                                delay=args.crawl_delay, sync=args.sync,
                                whisper_model=args.whisper_model, language=args.language)

    if args.sitemap:
        jobs += ingest_by_sitemap(args.sitemap, sync=args.sync,
                                  whisper_model=args.whisper_model, language=args.language)

    if args.pdf_urls:
        jobs += ingest_pdf_urls(args.pdf_urls, sync=args.sync, language=args.language)

    if args.pdfs:
        jobs += ingest_pdf_directory(args.pdfs, sync=args.sync, language=args.language)

    if args.urls:
        jobs += ingest_html_urls(args.urls, sync=args.sync, language=args.language)

    if args.video_urls:
        jobs += ingest_video_urls(args.video_urls, args.whisper_model, sync=args.sync,
                                  language=args.language)

    if args.videos:
        jobs += ingest_video_directory(args.videos, args.whisper_model, sync=args.sync,
                                       language=args.language)

    if jobs == 0:
        logger.warning("Nothing to ingest! Use --pdf-urls, --video-urls, or --urls.")
        sys.exit(1)

    if not args.sync:
        logger.info(f"\n{jobs} jobs queued. Monitor at: http://localhost:5555 (Flower)")
    else:
        logger.success(f"\nDone. {jobs} items processed.")


def ingest_by_crawl(
    start_url: str,
    max_pages: int | None = None,
    delay: float = 0.5,
    sync: bool = False,
    whisper_model: str = "base",
    language: str | None = None,
) -> int:
    """Discover URLs by crawling, then ingest each discovered type.

    A jw.org seed with a locale (``/tw/...``) limits the crawl to that locale's
    pages, so the language picker isn't followed into every other language.
    """
    path_prefix = _jw_path_prefix(start_url)
    if path_prefix:
        logger.info(f"Crawl limited to {path_prefix} pages")

    if sync:
        from ingestion.extractors.crawler import crawl_site

        urls = crawl_site(start_url, max_pages=max_pages, delay=delay, path_prefix=path_prefix)
        return _ingest_discovered_urls(urls, sync=sync, whisper_model=whisper_model,
                                       language=language)

    from ingestion.tasks.worker import crawl_and_enqueue

    parsed = urlparse(start_url)
    allowed_domain = parsed.netloc
    if not allowed_domain:
        logger.error(f"Invalid --crawl URL: {start_url}")
        return 0

    job_id = f"crawl-{uuid.uuid4().hex[:12]}"
    seed_url = _normalise_url(start_url)

    crawl_and_enqueue.delay(
        seed_url,
        job_id,
        allowed_domain,
        max_pages=max_pages,
        delay=delay,
        include_subdomains=False,
        respect_robots=True,
        use_playwright=True,
        path_prefix=path_prefix,
        language=language,
    )

    logger.info(
        f"Started async crawl job '{job_id}' from {seed_url} (max_pages={max_pages}, delay={delay}s)"
    )
    logger.info("Crawl dedupe key: crawl:{job_id}:seen (Redis set)")
    return 1


def ingest_by_sitemap(
    base_url: str,
    sync: bool = False,
    whisper_model: str = "base",
    language: str | None = None,
) -> int:
    """Discover URLs from sitemap.xml, then ingest each discovered type.
    A jw.org base URL with a locale keeps only that locale's URLs."""
    from ingestion.extractors.crawler import _in_path_scope, crawl_sitemap

    urls = crawl_sitemap(base_url)
    path_prefix = _jw_path_prefix(base_url)
    if path_prefix:
        before = len(urls)
        urls = [u for u in urls if _in_path_scope(urlparse(u).path, path_prefix)]
        logger.info(f"[FILTER] kept {len(urls)}/{before} sitemap URLs under {path_prefix}")
    return _ingest_discovered_urls(urls, sync=sync, whisper_model=whisper_model,
                                   language=language)


# ── URL-based ingestion ───────────────────────────────────────────────────────

def ingest_pdf_urls(urls_file: str, sync: bool = False, language: str | None = None) -> int:
    urls = _read_lines(urls_file)
    logger.info(f"Ingesting {len(urls)} PDF URLs from {urls_file}")
    for url in urls:
        if sync:
            _sync_pdf(url, language)
        else:
            from ingestion.tasks.worker import ingest_pdf
            ingest_pdf.delay(url, language=language)   # worker handles both paths and URLs
            logger.info(f"  Queued: {url}")
    return len(urls)


def ingest_video_urls(
    urls_file: str, model_size: str = "base", sync: bool = False, language: str | None = None
) -> int:
    urls = _read_lines(urls_file)
    logger.info(f"Ingesting {len(urls)} video URLs from {urls_file}")
    for url in urls:
        if sync:
            _sync_video(url, model_size, language)
        else:
            from ingestion.tasks.worker import ingest_video
            ingest_video.delay(url, is_url=True, model_size=model_size, language=language)
            logger.info(f"  Queued: {url}")
    return len(urls)


def ingest_html_urls(urls_file: str, sync: bool = False, language: str | None = None) -> int:
    from ingestion.extractors.crawler import _is_excluded_url

    urls = _read_lines(urls_file)
    urls = [u for u in urls if not _is_excluded_url(u)]
    logger.info(f"Ingesting {len(urls)} HTML page URLs from {urls_file}")
    for url in urls:
        if sync:
            _sync_html(url, language)
        else:
            from ingestion.tasks.worker import ingest_url
            ingest_url.delay(url, language=language)
            logger.info(f"  Queued: {url}")
    return len(urls)


# ── Local directory ingestion ─────────────────────────────────────────────────

def ingest_pdf_directory(directory: str, sync: bool = False, language: str | None = None) -> int:
    files = list(Path(directory).rglob("*.pdf"))
    logger.info(f"Found {len(files)} PDFs in {directory}")
    for f in files:
        if sync:
            _sync_pdf(str(f), language)
        else:
            from ingestion.tasks.worker import ingest_pdf
            ingest_pdf.delay(str(f), language=language)
            logger.info(f"  Queued: {f.name}")
    return len(files)


def ingest_video_directory(
    directory: str, model_size: str = "base", sync: bool = False, language: str | None = None
) -> int:
    extensions = {".mp4", ".mp3", ".wav", ".m4a", ".webm", ".ogg"}
    files = [f for f in Path(directory).rglob("*") if f.suffix.lower() in extensions]
    logger.info(f"Found {len(files)} video files in {directory}")
    for f in files:
        if sync:
            _sync_video(str(f), model_size, language)
        else:
            from ingestion.tasks.worker import ingest_video
            ingest_video.delay(str(f), is_url=False, model_size=model_size, language=language)
            logger.info(f"  Queued: {f.name}")
    return len(files)


def _ingest_discovered_urls(
    urls: list[str],
    sync: bool = False,
    whisper_model: str = "base",
    language: str | None = None,
) -> int:
    from ingestion.extractors.crawler import MEDIA_EXTENSIONS, _is_excluded_url

    before = len(urls)
    urls = [u for u in urls if not _is_excluded_url(u)]
    skipped = before - len(urls)
    if skipped:
        logger.info(f"[FILTER] skipped {skipped} excluded URLs (e.g. Bible pages)")

    pdf_urls = [u for u in urls if u.lower().endswith(".pdf")]
    video_urls = [u for u in urls if u.lower().endswith((".mp4", ".mp3", ".wav", ".m4a", ".webm", ".ogg"))]
    html_urls = [u for u in urls if _extension_from_url(u) not in MEDIA_EXTENSIONS]

    jobs = 0
    logger.info(
        f"Discovered URLs from crawl/sitemap: {len(html_urls)} HTML, {len(pdf_urls)} PDFs, {len(video_urls)} videos"
    )

    for url in html_urls:
        if sync:
            _sync_html(url, language)
        else:
            from ingestion.tasks.worker import ingest_url
            ingest_url.delay(url, language=language)
            logger.info(f"  Queued HTML: {url}")
        jobs += 1

    for url in pdf_urls:
        if sync:
            _sync_pdf(url, language)
        else:
            from ingestion.tasks.worker import ingest_pdf
            ingest_pdf.delay(url, language=language)
            logger.info(f"  Queued PDF: {url}")
        jobs += 1

    for url in video_urls:
        if sync:
            _sync_video(url, whisper_model, language)
        else:
            from ingestion.tasks.worker import ingest_video
            ingest_video.delay(url, is_url=True, model_size=whisper_model, language=language)
            logger.info(f"  Queued video: {url}")
        jobs += 1

    return jobs


# ── Sync (no Celery) helpers ──────────────────────────────────────────────────

def _sync_pdf(source: str, language: str | None = None):
    from ingestion.extractors.pdf_extractor import extract_pdf
    from ingestion.utils.chunker import chunk_pages
    from ingestion.utils.embedder import (
        embed_chunks, upsert_embeddings, init_collection, source_exists,
    )
    import os

    lang = languages.resolve_ingest_language(source, language).code
    client = init_collection()
    if source_exists(client, source, lang):
        logger.info(f"[PDF][sync] Skipped (already in Pinecone): {source}")
        return

    pages = extract_pdf(source)  # handles URLs automatically
    chunks = chunk_pages(pages)
    embedded = embed_chunks(
        chunks,
        api_key=os.getenv("GEMINI_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
    )
    upsert_embeddings(embedded, client, language=lang)


def _sync_html(url: str, language: str | None = None):
    from ingestion.extractors.html_scraper import scrape_url
    from ingestion.utils.chunker import chunk_text
    from ingestion.utils.embedder import (
        embed_chunks, upsert_embeddings, init_collection, source_exists,
    )
    from ingestion.utils.preprocessor import preprocess, enrich_chunks_with_context
    import os

    lang = languages.resolve_ingest_language(url, language).code
    client = init_collection()
    if source_exists(client, url, lang):
        logger.info(f"[HTML][sync] Skipped (already in Pinecone): {url}")
        return

    page = scrape_url(url, language=lang)
    if not page:
        logger.warning(f"[HTML][sync] Skipped: {url} (scrape failed or page deemed non-content/error)")
        return

    page = preprocess(page)
    chunks = chunk_text(page.text, source=url, metadata=page.metadata)
    chunks = enrich_chunks_with_context(chunks, title=page.title or url, full_text=page.text)
    embedded = embed_chunks(
        chunks,
        api_key=os.getenv("GEMINI_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
    )
    upsert_embeddings(embedded, client, language=lang)


def _sync_video(source: str, model_size: str = "base", language: str | None = None):
    from ingestion.extractors.video_transcriber import transcribe_file, transcribe_url
    from ingestion.utils.chunker import chunk_text
    from ingestion.utils.embedder import (
        embed_chunks, upsert_embeddings, init_collection, source_exists,
    )
    import os

    lang = languages.resolve_ingest_language(source, language).code
    client = init_collection()
    if source_exists(client, source, lang):
        logger.info(f"[VIDEO][sync] Skipped (already in Pinecone): {source}")
        return

    if source.startswith("http://") or source.startswith("https://"):
        result = transcribe_url(source, model_size)
    else:
        result = transcribe_file(source, model_size)

    if not result:
        return
    chunks = chunk_text(result.transcript, source=source, metadata=result.metadata)
    embedded = embed_chunks(
        chunks,
        api_key=os.getenv("GEMINI_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
    )
    upsert_embeddings(embedded, client, language=lang)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_lines(path: str) -> list[str]:
    """Read a text file, skipping blank lines and # comments."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def _jw_path_prefix(url: str) -> str | None:
    """"/tw/" for a jw.org URL under a registered locale, else None."""
    lang = languages.from_jw_url(url)
    return f"/{lang.jw_locale}/" if lang else None


def _extension_from_url(url: str) -> str:
    from urllib.parse import urlparse

    path = urlparse(url).path.lower()
    dot = path.rfind(".")
    return path[dot:] if dot != -1 else ""


def _normalise_url(url: str) -> str:
    """Canonicalize URL for stable crawl deduplication."""
    url, _ = urldefrag(url)
    parsed = urlparse(url)

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Drop default ports so canonical form is stable.
    if (scheme == "http" and netloc.endswith(":80")) or (scheme == "https" and netloc.endswith(":443")):
        netloc = netloc.rsplit(":", 1)[0]

    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")

    # Remove noisy tracking query params and sort remaining params.
    query_params = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        lowered = key.lower()
        if lowered.startswith("utm_") or lowered in {"gclid", "fbclid"}:
            continue
        query_params.append((key, value))
    query_params.sort()
    query = urlencode(query_params, doseq=True)

    return urlunparse((scheme, netloc, path, "", query, ""))


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
