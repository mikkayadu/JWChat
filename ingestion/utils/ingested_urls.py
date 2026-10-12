"""
Ingested URLs
─────────────
Keeps ingested_urls.txt listing every source (page URL, PDF, video) whose
vectors were written to Pinecone, one per line:

    # URLs ingested into Pinecone, oldest first
    https://www.jw.org/en/bible-teachings/questions/original-sin
    ...

Each ingestion only appends its sources, so recording one never rereads or
rewrites the file and the end shows what was ingested most recently. A source
ingested again is appended again; read_ingested_urls drops the repeats. The
header is a comment, so the file also works as input for ``pipeline.py
--urls``. To count the sources, or to list pages ingested before the file
existed, rebuild it from Pinecone:

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
_HEADER = "# URLs ingested into Pinecone, oldest first\n"


def ingested_urls_path() -> Path:
    """INGESTED_URLS_FILE, else ingested_urls.txt in the project root."""
    return Path(os.getenv("INGESTED_URLS_FILE") or _DEFAULT_PATH)


def read_ingested_urls(path: Path | None = None) -> list[str]:
    """The listed sources, in order, each once."""
    path = path or ingested_urls_path()
    return list(dict.fromkeys(_parse(path.read_text(encoding="utf-8")))) if path.exists() else []


def record_ingested(sources) -> None:
    """Append ``sources`` to the list.

    Never raises: the vectors are already stored, so failing to update the
    list must not fail the ingestion.
    """
    new = [s for s in dict.fromkeys(sources) if s]
    if not new:
        return
    path = ingested_urls_path()
    try:
        with _locked(path, "a") as fh:
            fh.writelines(f"{url}\n" for url in new)
    except Exception as e:
        logger.warning(f"Could not update {path}: {e}")


def write_ingested_urls(sources, path: Path | None = None) -> int:
    """Replace the list with ``sources`` (order kept, duplicates dropped).
    Returns the count."""
    urls = [s for s in dict.fromkeys(sources) if s]
    with _locked(path or ingested_urls_path(), "w") as fh:
        fh.write(_HEADER)
        fh.writelines(f"{url}\n" for url in urls)
    return len(urls)


@contextmanager
def _locked(path: Path, mode: str):
    """The file opened in ``mode`` under an exclusive lock: Celery runs several
    worker processes, and the lock keeps their lines from interleaving."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as fh:
        if fcntl:
            fcntl.flock(fh, fcntl.LOCK_EX)
        yield fh


def _parse(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
