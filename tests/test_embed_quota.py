from types import SimpleNamespace

import pytest
from google.api_core.exceptions import ResourceExhausted

from ingestion.tasks import worker
from ingestion.utils import embedder
from ingestion.utils.chunker import TextChunk
from ingestion.utils.embedder import EmbeddingQuotaExhausted, embed_chunks

_TEXT = "God loved the world so much that he gave his only-begotten Son so that everyone exercising faith"
_DAILY = ResourceExhausted(
    "You exceeded your current quota",
    details=['quota_id: "EmbedContentRequestsPerDayPerProjectPerModel-FreeTier"'],
)
_PER_MINUTE = ResourceExhausted(
    "You exceeded your current quota",
    details=['quota_id: "EmbedContentRequestsPerMinutePerProjectPerModel-FreeTier"',
             "retry_delay { seconds: 7 }"],
)


def _chunks():
    return [TextChunk(text=_TEXT, chunk_index=0, token_count=20, source="https://www.jw.org/en/x")]


@pytest.fixture
def sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr(embedder.time, "sleep", calls.append)
    monkeypatch.setattr("google.generativeai.configure", lambda **_: None)
    return calls


def _embed_responses(monkeypatch, *responses):
    queue = list(responses)

    def fake_embed(**_):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr("google.generativeai.embed_content", fake_embed)


def test_daily_quota_stops_immediately(monkeypatch, sleeps):
    _embed_responses(monkeypatch, _DAILY)
    with pytest.raises(EmbeddingQuotaExhausted):
        embed_chunks(_chunks(), api_key="k")
    assert sleeps == []


def test_per_minute_limit_waits_and_retries(monkeypatch, sleeps):
    _embed_responses(monkeypatch, _PER_MINUTE, {"embedding": [0.1] * 768})
    results = embed_chunks(_chunks(), api_key="k")
    assert len(results) == 1
    assert sleeps[0] == 8.0


def test_limit_that_never_clears_counts_as_exhausted(monkeypatch, sleeps):
    _embed_responses(monkeypatch, *[_PER_MINUTE] * (embedder._MAX_RATE_LIMIT_WAITS + 1))
    with pytest.raises(EmbeddingQuotaExhausted):
        embed_chunks(_chunks(), api_key="k")


def test_worker_requeues_page_and_shuts_down(monkeypatch):
    requeued, shutdowns = [], []
    monkeypatch.setattr(worker, "_already_ingested", lambda *_: False)
    monkeypatch.setattr("ingestion.extractors.html_scraper.scrape_url", lambda *_, **__: SimpleNamespace(text=_TEXT, title="T", metadata={}))
    monkeypatch.setattr("ingestion.utils.preprocessor.preprocess", lambda page: page)
    monkeypatch.setattr("ingestion.utils.preprocessor.enrich_chunks_with_context", lambda c, **_: c)
    monkeypatch.setattr("ingestion.utils.chunker.chunk_text", lambda *_, **__: _chunks())

    def quota_hit(*_):
        raise EmbeddingQuotaExhausted("daily")

    monkeypatch.setattr(worker, "_embed_and_store", quota_hit)
    monkeypatch.setattr(worker.ingest_url, "apply_async", lambda **kw: requeued.append(kw))
    monkeypatch.setattr(worker.app.control, "shutdown", lambda: shutdowns.append(True))

    result = worker.ingest_url("https://www.jw.org/en/x")
    assert result["status"] == "quota_exhausted"
    assert requeued == [{"args": ("https://www.jw.org/en/x",),
                         "kwargs": {"use_playwright": True, "language": None}}]
    assert shutdowns == [True]
