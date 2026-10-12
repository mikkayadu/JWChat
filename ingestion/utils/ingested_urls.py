"""
Ingested URLs
─────────────
Keeps ingested_urls.txt listing every source (page URL, PDF, video) whose
vectors were written to Pinecone, under a header line with the count:

    # 1,722 URLs ingested into Pinecone
    https://www.jw.org/en/bible-teachings/questions/original-sin
    ...

Each newly ingested source is appended once, so the end of the file shows
what was ingested most recently. The header is a comment, so the file also
works as input for ``pipeline.py --urls``. To list pages ingested before the
file existed, rebuild it from Pinecone:

    python -m scripts.ingest_stats --out ingested_urls.txt
"""

import os
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

try:
    import fcntl
except ImportError:  # Windows has no flock; --sync runs in one process anyway
    fcntl = None

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "ingested_urls.txt"


def ingested_urls_path() -> Path:
    """INGESTED_URLS_FILE, else ingested_urls.txt in the project root."""
    return Path(os.getenv("INGESTED_URLS_FILE") or _DEFAULT_PATH)


def read_ingested_urls(path: Path | None = None) -> list[str]:
    path = path or ingested_urls_path()
    return _parse(path.read_text(encoding="utf-8")) if path.exists() else []


def record_ingested(sources) -> None:
    """Append the sources that aren't listed yet and update the count.

    Never raises: the vectors are already stored, so failing to update the
    list must not fail the ingestion.
    """
    path = ingested_urls_path()
    try:
        with _locked(path) as fh:
            urls = _parse(fh.read())
            listed = set(urls)
            new = [s for s in dict.fromkeys(sources) if s and s not in listed]
            if not new:
                return
            _write(fh, urls + new)
        logger.info(f"[INGESTED] {len(urls) + len(new):,} URLs listed in {path.name}")
    except Exception as e:
        logger.warning(f"Could not update {path}: {e}")


def write_ingested_urls(sources, path: Path | None = None) -> int:
    """Replace the list with ``sources`` (order kept, duplicates dropped).
    Returns the count."""
    urls = [s for s in dict.fromkeys(sources) if s]
    with _locked(path or ingested_urls_path()) as fh:
        _write(fh, urls)
    return len(urls)


@contextmanager
def _locked(path: Path):
    """The file, opened for update under an exclusive lock: Celery runs
    several worker processes, and each one rewrites the file's count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    with open(path, "r+", encoding="utf-8") as fh:
        if fcntl:
            fcntl.flock(fh, fcntl.LOCK_EX)
        yield fh


def _parse(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def _write(fh, urls: list[str]) -> None:
    fh.seek(0)
    fh.truncate()
    fh.write(f"# {len(urls):,} URLs ingested into Pinecone\n")
    fh.writelines(f"{url}\n" for url in urls)
