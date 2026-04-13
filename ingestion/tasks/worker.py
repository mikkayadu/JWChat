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
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()

# ── App Setup ────────────────────────────────────────────────────────────────

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
    worker_prefetch_multiplier=1,  # don't grab too many tasks at once
    result_expires=3600,
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _get_vector_client():
    from ingestion.utils.embedder import init_collection
    return init_collection(
        collection_name=os.getenv("PINECONE_INDEX", "website-rag"),
        vector_size=int(os.getenv("EMBEDDING_DIMENSIONS", 768)),
    )


def _embed_and_store(chunks):
    from ingestion.utils.embedder import embed_chunks, upsert_embeddings

    if not chunks:
        return 0

    embedded = embed_chunks(
        chunks,
        api_key=os.getenv("GEMINI_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL", "models/embedding-001"),
    )
    client = _get_vector_client()
    upsert_embeddings(embedded, client, os.getenv("PINECONE_INDEX", "website-rag"))
    return len(embedded)


# ── Tasks ─────────────────────────────────────────────────────────────────────

@app.task(bind=True, max_retries=3, default_retry_delay=10)
def ingest_pdf(self, file_path: str):
    """Ingest a single PDF file."""
    try:
        from ingestion.extractors.pdf_extractor import extract_pdf
        from ingestion.utils.chunker import chunk_pages

        logger.info(f"[PDF] Starting: {file_path}")
        pages = extract_pdf(file_path)
        chunks = chunk_pages(
            pages,
            chunk_size=int(os.getenv("CHUNK_SIZE", 400)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 80)),
        )
        count = _embed_and_store(chunks)
        logger.success(f"[PDF] Done: {file_path} → {count} vectors")
        return {"file": file_path, "vectors": count}

    except Exception as e:
        logger.error(f"[PDF] Failed: {file_path}: {e}")
        raise self.retry(exc=e)


@app.task(bind=True, max_retries=3, default_retry_delay=10)
def ingest_url(self, url: str, use_playwright: bool = True):
    """Scrape and ingest a single HTML page."""
    try:
        from ingestion.extractors.html_scraper import scrape_url
        from ingestion.utils.chunker import chunk_text

        logger.info(f"[HTML] Starting: {url}")
        page = scrape_url(url, use_playwright=use_playwright)
        if not page:
            return {"url": url, "vectors": 0, "error": "scrape failed"}

        chunks = chunk_text(
            page.text,
            source=url,
            metadata=page.metadata,
            chunk_size=int(os.getenv("CHUNK_SIZE", 400)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 80)),
        )
        count = _embed_and_store(chunks)
        logger.success(f"[HTML] Done: {url} → {count} vectors")
        return {"url": url, "vectors": count}

    except Exception as e:
        logger.error(f"[HTML] Failed: {url}: {e}")
        raise self.retry(exc=e)


@app.task(bind=True, max_retries=2, default_retry_delay=30)
def ingest_video(self, source: str, is_url: bool = False, model_size: str = "base"):
    """Transcribe and ingest a video file or URL."""
    try:
        from ingestion.utils.chunker import chunk_text

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
        count = _embed_and_store(chunks)
        logger.success(f"[VIDEO] Done: {source} → {count} vectors")
        return {"source": source, "vectors": count}

    except Exception as e:
        logger.error(f"[VIDEO] Failed: {source}: {e}")
        raise self.retry(exc=e)
