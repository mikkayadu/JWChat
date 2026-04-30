---
source: Context7 API
library: Pinecone
package: pinecone
topic: upsert best practices metadata filtering and quality gates before embedding/upsert
fetched: 2026-04-22T00:00:00Z
official_docs: https://docs.pinecone.io/guides/index-data/upsert-data
---

## Key documented constraints and capabilities

- Upsert batch limits: **max 1000 vector records** or **2MB** per request.
- Text upsert (`upsert_records`) max: **96 records** or **2MB** per request.
- Max metadata size per record: **40KB**.
- Max record ID length: **512 chars**.
- Metadata filtering operators include `$eq`, `$in`, `$exists`, `$and`, `$or`, etc.
- Namespaces are the primary isolation mechanism for multi-tenant/index partitioning.

## Guard design for RAG ingestion (before embedding/upsert)

1. **Source-quality gate (pre-embedding):**
   - reject blocked/captcha/error pages from crawler stage.
   - reject empty/very short content.
   - reject duplicate canonical URLs/chunk hashes.

2. **Schema gate (metadata contract):**
   enforce required metadata fields on each chunk:
   - `source_url`, `canonical_url`, `document_id`, `chunk_id`, `ingested_at`, `content_type`, `quality_passed`.

3. **Size/format gate:**
   - ensure `id` <= 512 chars.
   - keep metadata < 40KB.
   - batch by both count and payload bytes (<= 1000 records and <= 2MB for vectors).

4. **Upsert gate:**
   - only upsert chunks with `quality_passed=True`.
   - set filterable fields explicitly for retrieval-time filtering.

## Python ingestion pattern

```python
from pinecone import Pinecone
from datetime import datetime, timezone
import json

pc = Pinecone(api_key="YOUR_API_KEY")
index = pc.Index(host="INDEX_HOST")

REQUIRED_META = {
    "source_url", "canonical_url", "document_id", "chunk_id",
    "ingested_at", "content_type", "quality_passed"
}

def approx_bytes(obj) -> int:
    return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))

def validate_record(rec):
    rid = rec["id"]
    md = rec.get("metadata", {})

    if len(rid) > 512:
        return False, "id_too_long"
    if not REQUIRED_META.issubset(md.keys()):
        return False, "missing_required_metadata"
    if md.get("quality_passed") is not True:
        return False, "quality_not_passed"
    if approx_bytes(md) > 40 * 1024:
        return False, "metadata_too_large"
    return True, None

def upsert_with_gates(namespace: str, candidate_vectors: list[dict]):
    accepted, rejected = [], []

    for rec in candidate_vectors:
        ok, reason = validate_record(rec)
        if ok:
            accepted.append(rec)
        else:
            rejected.append({"id": rec.get("id"), "reason": reason})

    # batch by both count and payload size
    batch, batch_bytes = [], 0
    for rec in accepted:
        rec_bytes = approx_bytes(rec)
        if len(batch) >= 1000 or (batch_bytes + rec_bytes) > 2 * 1024 * 1024:
            index.upsert(namespace=namespace, vectors=batch)
            batch, batch_bytes = [], 0
        batch.append(rec)
        batch_bytes += rec_bytes

    if batch:
        index.upsert(namespace=namespace, vectors=batch)

    return {"accepted": len(accepted), "rejected": rejected[:20]}

# Example retrieval filter to enforce quality at query time too
filter_criteria = {
    "$and": [
        {"quality_passed": {"$eq": True}},
        {"content_type": {"$in": ["docs", "kb", "manual"]}},
        {"blocked_page": {"$ne": True}}
    ]
}
```

## Recommended operational best practices

- Use deterministic IDs (e.g., `{document_id}#{chunk_number}`) for idempotent re-runs.
- Use document-level delete by metadata filter before re-upserting changed docs.
- Keep noisy raw HTML out of metadata if near limits; store only retrieval-critical fields.
- Enforce quality twice: pre-upsert (pipeline) and query-time (metadata filter).

## Sources used

- https://docs.pinecone.io/guides/index-data/upsert-data
- https://docs.pinecone.io/reference/api/database-limits
- https://docs.pinecone.io/guides/search/filter-by-metadata
- https://docs.pinecone.io/guides/index-data/data-modeling
- https://docs.pinecone.io/guides/index-data/implement-multitenancy
