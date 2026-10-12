"""
Ingestion stats
───────────────
Counts how many distinct sources (URLs, PDFs, videos) are in Pinecone, per
language namespace. Pinecone can't count distinct metadata values, so this
lists every vector ID, fetches the metadata in batches and tallies ``source``.
Read-only.

Usage:
    python -m scripts.ingest_stats                  # every registered language
    python -m scripts.ingest_stats --language tw
    python -m scripts.ingest_stats --out ingested_urls.txt   # also save the list
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import languages  # noqa: E402
from ingestion.utils.ingested_urls import write_ingested_urls  # noqa: E402

load_dotenv()

FETCH_BATCH = 100  # fetch is a GET; 100 UUIDs keeps the URL well under limits


def count_sources(index, namespace: str) -> Counter:
    """Chunks per source in ``namespace``."""
    per_source: Counter = Counter()
    for id_page in index.list(namespace=namespace):
        for i in range(0, len(id_page), FETCH_BATCH):
            fetched = index.fetch(ids=id_page[i : i + FETCH_BATCH], namespace=namespace)
            for vector in fetched.vectors.values():
                per_source[(vector.metadata or {}).get("source", "<no source>")] += 1
    return per_source


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", help="Only this language code (default: all registered)")
    parser.add_argument("--out", help="Write every ingested source, one per line, to this file")
    args = parser.parse_args()

    from pinecone import Pinecone

    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(
        os.getenv("PINECONE_INDEX", "website-rag")
    )
    langs = [languages.get(args.language)] if args.language else list(languages.LANGUAGES.values())

    all_sources: list[str] = []
    for lang in langs:
        per_source = count_sources(index, lang.namespace)
        chunks = sum(per_source.values())
        print(f"{lang.name} (namespace '{lang.namespace}'): "
              f"{len(per_source)} sources, {chunks} chunks", end="")
        if per_source:
            print(f", {chunks / len(per_source):.1f} chunks per source on average")
            kinds = Counter(
                "pdf" if s.lower().endswith(".pdf") else "web" if s.startswith("http") else "file"
                for s in per_source
            )
            print("  " + ", ".join(f"{n} {kind}" for kind, n in kinds.most_common()))
        else:
            print()
        all_sources += sorted(per_source)

    if args.out:
        # Same format as the list ingestion keeps up to date (count header, one
        # source per line), so --out ingested_urls.txt rebuilds that list.
        count = write_ingested_urls(all_sources, Path(args.out))
        print(f"Wrote {count} sources to {args.out}")


if __name__ == "__main__":
    main()
