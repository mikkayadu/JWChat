"""
Embedder + Vector Store
────────────────────────
1. Calls Gemini embedding API in batches
2. Upserts vectors into Pinecone with full metadata
"""

import os
import unicodedata
import time
import uuid
import re
from loguru import logger
from ingestion.utils.chunker import TextChunk


_ERROR_TEXT_MARKERS = [
    "error: the request could not be satisfied",
    "cloudfront",
    "access denied",
    "forbidden",
    "request blocked",
    "verify you are human",
    "captcha",
]


# Per-minute rate limits clear on their own; this many waits in a row without
# one going through means the quota is spent (e.g. a billing cap), not busy.
_MAX_RATE_LIMIT_WAITS = 5


class EmbeddingQuotaExhausted(RuntimeError):
    """Gemini refused to embed because the quota is used up. Retrying before
    it resets only wastes work, so ingestion should stop."""


def _is_rate_limit_error(e: Exception) -> bool:
    from google.api_core.exceptions import TooManyRequests

    return isinstance(e, TooManyRequests) or "RESOURCE_EXHAUSTED" in str(e)


def _retry_delay_seconds(e: Exception) -> float:
    """The wait Gemini suggests in a 429 ("retry_delay { seconds: 22 }" or
    "Please retry in 22.3s"), else a minute."""
    text = str(e)
    match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", text) or re.search(
        r"retry in ([\d.]+)s", text
    )
    return float(match.group(1)) + 1 if match else 60.0


def _embed_content(genai, model: str, text: str):
    """genai.embed_content, waiting out per-minute rate limits. Raises
    EmbeddingQuotaExhausted when a daily quota is spent, or when the limit
    doesn't clear after several waits."""
    for waits in range(_MAX_RATE_LIMIT_WAITS + 1):
        try:
            return genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document",
                output_dimensionality=768,
            )
        except Exception as e:
            if not _is_rate_limit_error(e):
                raise
            if "PerDay" in str(e) or waits == _MAX_RATE_LIMIT_WAITS:
                raise EmbeddingQuotaExhausted(str(e)) from e
            delay = _retry_delay_seconds(e)
            logger.warning(f"Gemini rate limit hit — waiting {delay:.0f}s before retrying")
            time.sleep(delay)


def _is_low_quality_error_chunk(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return True

    if any(marker in normalized for marker in _ERROR_TEXT_MARKERS):
        return True

    word_count = len(re.findall(r"\b\w+\b", normalized))
    return word_count < 12


def _normalize_metadata_text(text: str) -> str:
    """Tidy chunk text for storage in Pinecone metadata.

    Folds curly quotes and dashes and strips zero-width characters, but keeps
    every other Unicode character: Twi's ɛ/ɔ (and Greek/Hebrew terms in study
    notes) must survive into the LLM context and UI snippets. NFC rather than
    NFKC so compatibility characters are left as the source wrote them.
    """
    if not text:
        return ""

    normalized = (
        text.replace("“", "\"")
        .replace("”", "\"")
        .replace("‘", "'")
        .replace("’", "'")
        .replace("—", "-")
        .replace("–", "-")
        .replace("\u200b", "")
        .replace("\ufeff", "")
    )

    return unicodedata.normalize("NFC", normalized)


def normalize_vector_dimensions(vector: list[float], target_dim: int) -> list[float]:
    """Resize a vector to match the target dimension for Pinecone compatibility."""
    if target_dim <= 0:
        return vector

    current_dim = len(vector)
    if current_dim == target_dim:
        return vector
    if current_dim > target_dim:
        return vector[:target_dim]
    return vector + [0.0] * (target_dim - current_dim)


def embed_chunks(
    chunks: list[TextChunk],
    api_key: str | None = None,
    model: str = "models/gemini-embedding-001",
    batch_size: int = 100,
) -> list[dict]:
    """
    Embed a list of TextChunks using Gemini.
    Returns list of dicts: {chunk, vector}
    """
    import google.generativeai as genai

    if not chunks:
        return []

    resolved_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_key:
        raise ValueError("Missing GEMINI_API_KEY for embeddings")

    genai.configure(api_key=resolved_key)
    results: list[dict] = []
    resize_logged = False

    try:
        target_dim = int(os.getenv("EMBEDDING_DIMENSIONS", "0"))
    except ValueError:
        target_dim = 0

    # Filter low-quality chunks BEFORE calling the API — saves embedding cost
    valid_chunks = [c for c in chunks if not _is_low_quality_error_chunk(c.text)]
    skipped = len(chunks) - len(valid_chunks)
    if skipped:
        logger.warning(f"Pre-embed quality filter: skipped {skipped} low-quality chunks")
    chunks = valid_chunks

    if not chunks:
        logger.warning("No valid chunks remain after quality filter")
        return results

    logger.info(f"Embedding {len(chunks)} chunks in batches of {batch_size}...")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        for chunk in batch:
            success = False
            for attempt in range(3):
                try:
                    response = _embed_content(genai, model, chunk.text)
                    vector = response["embedding"]
                    if target_dim > 0 and len(vector) != target_dim:
                        if not resize_logged:
                            logger.warning(
                                f"Resizing embedding vectors from {len(vector)} to {target_dim} to match configured index dimension"
                            )
                            resize_logged = True
                        vector = normalize_vector_dimensions(vector, target_dim)
                    results.append({"chunk": chunk, "vector": vector})
                    success = True
                    break
                except EmbeddingQuotaExhausted:
                    # Drop the whole page rather than store part of it: the
                    # skip-if-ingested check would never fill in the rest.
                    raise
                except Exception as e:
                    wait = 2 ** attempt
                    logger.warning(
                        f"Embedding failed for chunk {chunk.chunk_index} (attempt {attempt + 1}): {e}. Retrying in {wait}s..."
                    )
                    time.sleep(wait)

            if not success:
                logger.error(f"Skipping chunk {chunk.chunk_index} after 3 failed attempts")

        time.sleep(0.05)

    logger.success(f"Embedded {len(results)}/{len(chunks)} chunks")
    return results


def init_collection(
    host: str = "",
    port: int = 0,
    collection_name: str = "website-rag",
    vector_size: int = 768,
):
    """
    Initialize a Pinecone index (creates it if missing).
    host/port are accepted for backward compatibility but ignored.
    """
    from pinecone import Pinecone, ServerlessSpec

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("Missing PINECONE_API_KEY")

    index_name = os.getenv("PINECONE_INDEX", collection_name)
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")
    dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", vector_size))

    pc = Pinecone(api_key=api_key)
    existing = [idx["name"] for idx in pc.list_indexes()]

    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dimensions,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        logger.success(f"Created Pinecone index: {index_name}")
    else:
        logger.info(f"Pinecone index '{index_name}' already exists")

    return pc.Index(index_name)


def upsert_embeddings(
    embedded: list[dict],
    client,
    collection_name: str = "website-rag",
    batch_size: int = 100,
    language: str = "en",
):
    """
    Upsert embedded chunks into Pinecone, into ``language``'s namespace.
    Every vector's metadata also records its ``language``.
    """
    import languages

    lang = languages.get(language)
    if not embedded:
        logger.warning("No embedded chunks to upsert")
        return

    filtered_embedded = [
        item for item in embedded
        if not _is_low_quality_error_chunk(item["chunk"].text)
    ]
    skipped = len(embedded) - len(filtered_embedded)

    if skipped > 0:
        logger.warning(f"Skipped {skipped} low-quality/error chunks before upsert")

    if not filtered_embedded:
        logger.warning("No valid embedded chunks to upsert after quality filter")
        return

    logger.info(
        f"Upserting {len(filtered_embedded)} vectors into Pinecone "
        f"(language={lang.code}, namespace='{lang.namespace}')..."
    )
    # English stays in the legacy default namespace: omit the kwarg rather than
    # pass "" so the existing upsert call is unchanged for it.
    namespace_kwargs = {"namespace": lang.namespace} if lang.namespace else {}

    for i in range(0, len(filtered_embedded), batch_size):
        batch = filtered_embedded[i : i + batch_size]
        vectors = []

        for item in batch:
            chunk: TextChunk = item["chunk"]
            vector: list[float] = item["vector"]

            point_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{chunk.source}::chunk_{chunk.chunk_index}"
            ))

            metadata = {
                "text": _normalize_metadata_text(chunk.text),
                "source": chunk.source,
                **chunk.metadata,
                "language": lang.code,
            }

            vectors.append({"id": point_id, "values": vector, "metadata": metadata})

        client.upsert(vectors=vectors, **namespace_kwargs)
        logger.debug(f"Upserted batch {i // batch_size + 1} ({len(batch)} points)")

    logger.success(f"Done: {len(filtered_embedded)} vectors in '{collection_name}'")


def source_exists(client, source: str, language: str = "en") -> bool:
    """
    True if any vector from ``source`` is already in ``language``'s namespace,
    so callers can skip scraping and embedding it again. A failed lookup
    returns False: ingesting twice is safe (IDs are deterministic).
    """
    import languages

    lang = languages.get(language)
    namespace_kwargs = {"namespace": lang.namespace} if lang.namespace else {}
    dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", 768))
    # Only the metadata filter matters; any non-zero vector of the index size works.
    probe = [1.0] + [0.0] * (dimensions - 1)

    try:
        result = client.query(
            vector=probe,
            top_k=1,
            filter={"source": {"$eq": source}},
            include_values=False,
            include_metadata=False,
            **namespace_kwargs,
        )
    except Exception as e:
        logger.warning(f"Pinecone lookup failed for {source}: {e} — ingesting anyway")
        return False

    return bool(result.matches)
