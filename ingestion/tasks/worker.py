"""
Celery Worker
─────────────
Defines async tasks for parallel ingestion of thousands of files.
Each file type gets its own task so they can scale independently.

Start the worker with:
  celery -A worker worker --loglevel=info --concurrency=4

Then submit jobs from pipeline.py or any Python script.
"""

from celery import Celery
from kombu import Queue
from loguru import logger
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Celery puts the working directory on sys.path only while it imports this
# module, so pool processes later fail on top-level modules like `languages`.
# Append unconditionally: during that import the root is already on the path
# (Celery's temporary entry), and a "not in sys.path" guard would skip it.
sys.path.append(str(Path(__file__).resolve().parents[2]))

from ingestion.utils.embedder import EmbeddingQuotaExhausted

load_dotenv()

# billiard starts pool processes with "spawn" on macOS, and Celery only sets up
# the per-process task state that spawned children need when this is set —
# without it every task fails with "not enough values to unpack (expected 3,
# got 0)". Pool processes inherit it from the main process.
os.environ.setdefault("FORKED_BY_MULTIPROCESSING", "1")

# ── App Setup ────────────────────────────────────────────────────────────────

# Kombu's Redis transport keeps consumed queues in a set, so declared order is
# lost; the "sorted" strategy polls them alphabetically instead, and the "0-"
# prefix sorts this queue ahead of the "celery" crawl queue.
INGEST_QUEUE = "0-ingest"

app = Celery(
    "rag_ingestion",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,           # re-queue if worker crashes mid-task
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # don't grab too many tasks at once
    # Ingest tasks get their own queue, checked before the crawl queue. Each
    # crawled page queues ~20 crawl tasks but one ingest task, so in a shared
    # queue pages waited behind thousands of crawl tasks.
    task_queues=(Queue(INGEST_QUEUE), Queue("celery")),
    task_routes={
        "ingestion.tasks.worker.ingest_pdf": {"queue": INGEST_QUEUE},
        "ingestion.tasks.worker.ingest_url": {"queue": INGEST_QUEUE},
        "ingestion.tasks.worker.ingest_video": {"queue": INGEST_QUEUE},
    },
    broker_transport_options={"visibility_timeout": 3600, "queue_order_strategy": "sorted"},
    result_expires=3600,
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _get_vector_client():
    from ingestion.utils.embedder import init_collection
    return init_collection(
        collection_name=os.getenv("PINECONE_INDEX", "website-rag"),
        vector_size=int(os.getenv("EMBEDDING_DIMENSIONS", 768)),
    )


def _get_redis_client():
    import redis

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis.Redis.from_url(redis_url, decode_responses=True)


def _crawl_seen_key(job_id: str) -> str:
    return f"crawl:{job_id}:seen"


def _crawl_pages_key(job_id: str) -> str:
    return f"crawl:{job_id}:pages"


def _crawl_failed_key(job_id: str) -> str:
    return f"crawl:{job_id}:failed"


def _crawl_queued_key(job_id: str) -> str:
    return f"crawl:{job_id}:queued"


def _ttl_seconds() -> int:
    return int(os.getenv("CRAWL_JOB_TTL_SECONDS", "86400"))


def _embed_and_store(chunks, language: str = "en"):
    from ingestion.utils.embedder import embed_chunks, upsert_embeddings

    if not chunks:
        return 0

    embedded = embed_chunks(
        chunks,
        api_key=os.getenv("GEMINI_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
    )
    client = _get_vector_client()
    upsert_embeddings(
        embedded, client, os.getenv("PINECONE_INDEX", "website-rag"), language=language
    )
    return len(embedded)


def _pause_for_quota(task, *args, **kwargs):
    """Gemini's embedding quota is spent: put this item back on the queue and
    shut the worker down. Everything still queued waits in Redis, so starting
    the worker again once the quota resets carries on where this stopped."""
    task.apply_async(args=args, kwargs=kwargs)
    logger.error(
        "Gemini embedding quota exhausted — shutting down the worker. Queued jobs "
        "are kept in Redis; restart the worker after the quota resets "
        "(daily limits reset at midnight Pacific time)."
    )
    app.control.shutdown()


def _already_ingested(source: str, language: str) -> bool:
    from ingestion.utils.embedder import source_exists

    return source_exists(_get_vector_client(), source, language)


def _resolve_language(source: str, language: str | None) -> str:
    """Language code for a document: the explicit override wins, then the
    jw.org URL locale, then English."""
    import languages

    return languages.resolve_ingest_language(source, language).code


def _queue_media_ingestion(url: str, language: str | None = None):
    lowered = url.lower()
    if lowered.endswith(".pdf"):
        ingest_pdf.delay(url, language=language)
        return "pdf"

    if lowered.endswith((".mp4", ".mp3", ".wav", ".m4a", ".webm", ".ogg")):
        ingest_video.delay(
            url, is_url=True, model_size=os.getenv("WHISPER_MODEL", "base"), language=language
        )
        return "video"

    return "unknown"


# ── Tasks ─────────────────────────────────────────────────────────────────────

@app.task(bind=True, max_retries=3, default_retry_delay=10)
def ingest_pdf(self, file_path: str, language: str | None = None):
    """Ingest a single PDF file. ``language`` overrides the URL's language."""
    try:
        from ingestion.extractors.pdf_extractor import extract_pdf
        from ingestion.utils.chunker import chunk_pages

        lang = _resolve_language(file_path, language)
        if _already_ingested(file_path, lang):
            logger.info(f"[PDF] Skipped (already in Pinecone): {file_path}")
            return {"file": file_path, "vectors": 0, "status": "already_ingested"}

        logger.info(f"[PDF] Starting: {file_path}")
        pages = extract_pdf(file_path)
        chunks = chunk_pages(
            pages,
            chunk_size=int(os.getenv("CHUNK_SIZE", 400)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 80)),
        )
        count = _embed_and_store(chunks, lang)
        logger.success(f"[PDF] Done: {file_path} → {count} vectors")
        return {"file": file_path, "vectors": count}

    except EmbeddingQuotaExhausted:
        _pause_for_quota(ingest_pdf, file_path, language=language)
        return {"file": file_path, "vectors": 0, "status": "quota_exhausted"}

    except Exception as e:
        logger.error(f"[PDF] Failed: {file_path}: {e}")
        raise self.retry(exc=e)


@app.task(bind=True, max_retries=3, default_retry_delay=10)
def ingest_url(self, url: str, use_playwright: bool = True, language: str | None = None):
    """Scrape and ingest a single HTML page. ``language`` overrides the URL's language."""
    try:
        from ingestion.extractors.crawler import _is_excluded_url
        from ingestion.extractors.html_scraper import scrape_url
        from ingestion.utils.chunker import chunk_text
        from ingestion.utils.preprocessor import preprocess, enrich_chunks_with_context

        if _is_excluded_url(url):
            logger.info(f"[HTML] Skipped (excluded): {url}")
            return {"url": url, "vectors": 0, "error": "excluded_path"}

        lang = _resolve_language(url, language)
        if _already_ingested(url, lang):
            logger.info(f"[HTML] Skipped (already in Pinecone): {url}")
            return {"url": url, "vectors": 0, "status": "already_ingested"}

        logger.info(f"[HTML] Starting: {url} (language={lang})")
        page = scrape_url(url, use_playwright=use_playwright, language=lang)
        if not page:
            logger.warning(f"[HTML] Skipped: {url} (scrape failed or page deemed non-content/error)")
            return {"url": url, "vectors": 0, "error": "invalid_or_error_page"}

        page = preprocess(page)
        chunks = chunk_text(
            page.text,
            source=url,
            metadata=page.metadata,
            chunk_size=int(os.getenv("CHUNK_SIZE", 400)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 80)),
        )
        chunks = enrich_chunks_with_context(chunks, title=page.title or url, full_text=page.text)
        count = _embed_and_store(chunks, lang)
        logger.success(f"[HTML] Done: {url} → {count} vectors")
        return {"url": url, "vectors": count}

    except EmbeddingQuotaExhausted:
        _pause_for_quota(ingest_url, url, use_playwright=use_playwright, language=language)
        return {"url": url, "vectors": 0, "status": "quota_exhausted"}

    except Exception as e:
        logger.error(f"[HTML] Failed: {url}: {e}")
        raise self.retry(exc=e)


@app.task(bind=True, max_retries=2, default_retry_delay=30)
def ingest_video(
    self,
    source: str,
    is_url: bool = False,
    model_size: str = "base",
    language: str | None = None,
):
    """Transcribe and ingest a video file or URL. ``language`` overrides the URL's language."""
    try:
        from ingestion.utils.chunker import chunk_text

        lang = _resolve_language(source, language)
        if _already_ingested(source, lang):
            logger.info(f"[VIDEO] Skipped (already in Pinecone): {source}")
            return {"source": source, "vectors": 0, "status": "already_ingested"}

        logger.info(f"[VIDEO] Starting: {source}")

        if is_url:
            from ingestion.extractors.video_transcriber import transcribe_url
            result = transcribe_url(source, model_size=model_size)
        else:
            from ingestion.extractors.video_transcriber import transcribe_file
            result = transcribe_file(source, model_size=model_size)

        if not result:
            return {"source": source, "vectors": 0, "error": "transcription failed"}

        chunks = chunk_text(
            result.transcript,
            source=source,
            metadata=result.metadata,
            chunk_size=int(os.getenv("CHUNK_SIZE", 400)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 80)),
        )
        count = _embed_and_store(chunks, lang)
        logger.success(f"[VIDEO] Done: {source} → {count} vectors")
        return {"source": source, "vectors": count}

    except EmbeddingQuotaExhausted:
        _pause_for_quota(
            ingest_video, source, is_url=is_url, model_size=model_size, language=language
        )
        return {"source": source, "vectors": 0, "status": "quota_exhausted"}

    except Exception as e:
        logger.error(f"[VIDEO] Failed: {source}: {e}")
        raise self.retry(exc=e)


@app.task(bind=True, max_retries=5, default_retry_delay=5)
def crawl_and_enqueue(
    self,
    url: str,
    job_id: str,
    allowed_domain: str,
    max_pages: int | None = None,
    delay: float = 0.0,
    include_subdomains: bool = False,
    respect_robots: bool = True,
    use_playwright: bool = True,
    path_prefix: str | None = None,
    language: str | None = None,
):
    """
    Crawl one URL, queue ingestion for it, and fan out child crawl tasks.
    URL dedupe is atomic via Redis SADD on crawl:{job_id}:seen.
    ``path_prefix`` (e.g. "/tw/") keeps the crawl inside one language, and
    ``language`` overrides the language each page would get from its URL.
    """
    try:
        from urllib.parse import urlparse
        from ingestion.extractors.crawler import (
            MEDIA_EXTENSIONS,
            SKIP_EXTENSIONS,
            _extract_links,
            _fetch,
            _get_extension,
            _is_disallowed,
            _is_video_page_url,
            _load_robots,
            _normalise,
        )

        redis_client = _get_redis_client()
        seen_key = _crawl_seen_key(job_id)
        pages_key = _crawl_pages_key(job_id)
        failed_key = _crawl_failed_key(job_id)
        ttl_seconds = _ttl_seconds()

        # Video platform pages (e.g. JW.org hash-routed URLs) must be detected
        # before _normalise strips their fragment — the fragment IS the video id.
        if _is_video_page_url(url):
            is_new = redis_client.sadd(seen_key, url)
            if is_new == 0:
                return {"url": url, "status": "duplicate"}
            redis_client.expire(seen_key, ttl_seconds)
            redis_client.expire(pages_key, ttl_seconds)
            redis_client.expire(failed_key, ttl_seconds)
            ingest_video.delay(
                url, is_url=True, model_size=os.getenv("WHISPER_MODEL", "base"), language=language
            )
            logger.info(f"[CRAWL] Queued video page: {url}")
            return {"url": url, "status": "queued_video_page", "type": "video"}

        canonical_url = _normalise(url)

        # Atomic dedupe gate: only first task for a URL proceeds.
        is_new = redis_client.sadd(seen_key, canonical_url)
        if is_new == 0:
            return {"url": canonical_url, "status": "duplicate"}

        redis_client.expire(seen_key, ttl_seconds)
        redis_client.expire(pages_key, ttl_seconds)
        redis_client.expire(failed_key, ttl_seconds)

        ext = _get_extension(canonical_url)
        if ext in MEDIA_EXTENSIONS:
            media_type = _queue_media_ingestion(canonical_url, language)
            logger.info(f"[CRAWL] Queued {media_type}: {canonical_url}")
            return {"url": canonical_url, "status": "queued_media", "type": media_type}

        if ext in SKIP_EXTENSIONS:
            return {"url": canonical_url, "status": "skipped_extension"}

        current_count = int(redis_client.get(pages_key) or 0)
        if max_pages is not None and current_count >= max_pages:
            return {"url": canonical_url, "status": "max_pages_reached", "pages": current_count}

        if respect_robots:
            scheme = urlparse(canonical_url).scheme or "https"
            base_url = f"{scheme}://{allowed_domain}"
            disallowed = _load_robots(base_url)
            if _is_disallowed(canonical_url, disallowed):
                return {"url": canonical_url, "status": "blocked_by_robots"}

        if delay > 0:
            time.sleep(delay)

        html = _fetch(canonical_url)
        if not html:
            redis_client.incr(failed_key)
            return {"url": canonical_url, "status": "fetch_failed"}

        pages = redis_client.incr(pages_key)
        if max_pages is not None and pages > max_pages:
            return {"url": canonical_url, "status": "max_pages_reached", "pages": pages}

        ingest_url.delay(canonical_url, use_playwright=use_playwright, language=language)

        links = _extract_links(
            html,
            canonical_url,
            allowed_domain,
            include_subdomains,
            path_prefix,
        )

        queued_key = _crawl_queued_key(job_id)
        queued_children = 0
        for link in links:
            # Pages share most of their links (navigation, footers), so queue
            # each URL once per job: deduping only when a task runs left ~175
            # duplicate tasks per crawled page sitting in Redis.
            dedupe_key = link if _is_video_page_url(link) else _normalise(link)
            if not redis_client.sadd(queued_key, dedupe_key):
                continue
            crawl_and_enqueue.delay(
                link,
                job_id,
                allowed_domain,
                max_pages=max_pages,
                delay=delay,
                include_subdomains=include_subdomains,
                respect_robots=respect_robots,
                use_playwright=use_playwright,
                path_prefix=path_prefix,
                language=language,
            )
            queued_children += 1
        redis_client.expire(queued_key, ttl_seconds)

        logger.info(
            f"[CRAWL] {canonical_url} | pages={pages}/{max_pages or '∞'} | children_queued={queued_children}"
        )
        return {
            "url": canonical_url,
            "status": "queued",
            "pages": pages,
            "children_queued": queued_children,
        }

    except Exception as e:
        logger.error(f"[CRAWL] Failed: {url}: {e}")
        raise self.retry(exc=e)
