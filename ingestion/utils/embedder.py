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


def _is_low_quality_error_chunk(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return True

    if any(marker in normalized for marker in _ERROR_TEXT_MARKERS):
        return True

    word_count = len(re.findall(r"\b\w+\b", normalized))
    return word_count < 12


def _normalize_metadata_text(text: str) -> str:
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

    normalized = unicodedata.normalize("NFKC", normalized)
    return normalized.encode("latin-1", "ignore").decode("latin-1")


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
    model: str = "models/embedding-001",
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

    logger.info(f"Embedding {len(chunks)} chunks in batches of {batch_size}...")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        for chunk in batch:
            success = False
            for attempt in range(3):
                try:
                    response = genai.embed_content(
                        model=model,
                        content=chunk.text,
                        task_type="retrieval_document",
                    )
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
):
    """
    Upsert embedded chunks into Pinecone.
    """
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

    logger.info(f"Upserting {len(filtered_embedded)} vectors into Pinecone...")

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
            }

            vectors.append({"id": point_id, "values": vector, "metadata": metadata})

        client.upsert(vectors=vectors)
        logger.debug(f"Upserted batch {i // batch_size + 1} ({len(batch)} points)")

    logger.success(f"Done: {len(filtered_embedded)} vectors in '{collection_name}'")
