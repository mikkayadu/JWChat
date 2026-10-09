import pytest

from ingestion.utils import embedder
from ingestion.utils.chunker import TextChunk
from ingestion.utils.embedder import embed_chunks

_TEXT = "God loved the world so much that he gave his only-begotten Son so that everyone exercising faith"


def _chunks(n):
    return [TextChunk(text=_TEXT, chunk_index=i, token_count=20, source="https://www.jw.org/en/x") for i in range(n)]


@pytest.fixture(autouse=True)
def no_waits(monkeypatch):
    monkeypatch.setattr(embedder.time, "sleep", lambda _: None)
    monkeypatch.setattr("google.generativeai.configure", lambda **_: None)


def test_page_is_embedded_in_one_request(monkeypatch):
    requests = []

    def fake_embed(**kwargs):
        requests.append(kwargs["content"])
        return {"embedding": [[0.1] * 768 for _ in kwargs["content"]]}

    monkeypatch.setattr("google.generativeai.embed_content", fake_embed)
    results = embed_chunks(_chunks(3), api_key="k")
    assert len(requests) == 1 and len(requests[0]) == 3
    assert [r["chunk"].chunk_index for r in results] == [0, 1, 2]


def test_failing_batch_drops_the_whole_page(monkeypatch):
    def fake_embed(**kwargs):
        # The first batch (2 chunks) succeeds; the second (1 chunk) keeps failing.
        if len(kwargs["content"]) == 1:
            raise RuntimeError("server error")
        return {"embedding": [[0.1] * 768 for _ in kwargs["content"]]}

    monkeypatch.setattr("google.generativeai.embed_content", fake_embed)
    assert embed_chunks(_chunks(3), api_key="k", batch_size=2) == []
