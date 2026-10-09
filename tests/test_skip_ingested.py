from types import SimpleNamespace

from ingestion.tasks import worker
from ingestion.utils.embedder import source_exists

_URL = "https://www.jw.org/en/library/magazines/x/"


class _FakeIndex:
    def __init__(self, matches=(), error=None):
        self.matches = list(matches)
        self.error = error
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(matches=self.matches)


def test_source_found_when_a_vector_matches():
    index = _FakeIndex(matches=[{"id": "a"}])
    assert source_exists(index, _URL) is True
    assert index.calls[0]["filter"] == {"source": {"$eq": _URL}}
    assert "namespace" not in index.calls[0]


def test_source_missing_when_no_vector_matches():
    assert source_exists(_FakeIndex(), _URL) is False


def test_twi_lookup_uses_twi_namespace():
    index = _FakeIndex()
    source_exists(index, "https://www.jw.org/tw/x", language="tw")
    assert index.calls[0]["namespace"] == "tw"


def test_failed_lookup_lets_ingestion_proceed():
    assert source_exists(_FakeIndex(error=RuntimeError("down")), _URL) is False


def test_ingest_url_skips_scraping_when_already_stored(monkeypatch):
    monkeypatch.setattr(worker, "_get_vector_client", lambda: _FakeIndex(matches=[{"id": "a"}]))

    def fail(*_, **__):
        raise AssertionError("an already-ingested page must not be scraped")

    monkeypatch.setattr("ingestion.extractors.html_scraper.scrape_url", fail)
    result = worker.ingest_url(_URL)
    assert result == {"url": _URL, "vectors": 0, "status": "already_ingested"}
