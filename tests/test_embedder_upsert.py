from ingestion.utils.chunker import TextChunk
from ingestion.utils.embedder import upsert_embeddings

_TEXT = "Na Onyankopɔn dɔ wiase paa ma enti ɔde ne Ba a ɔwoo no koro no mae sɛnea ɛbɛyɛ"


class _FakeIndex:
    def __init__(self):
        self.calls = []

    def upsert(self, vectors, **kwargs):
        self.calls.append((vectors, kwargs))


def _embedded():
    chunk = TextChunk(text=_TEXT, chunk_index=0, token_count=20, source="https://www.jw.org/tw/x")
    return [{"chunk": chunk, "vector": [0.1, 0.2]}]


def test_twi_upserts_into_twi_namespace():
    index = _FakeIndex()
    upsert_embeddings(_embedded(), index, language="tw")
    vectors, kwargs = index.calls[0]
    assert kwargs == {"namespace": "tw"}
    assert vectors[0]["metadata"]["language"] == "tw"
    assert vectors[0]["metadata"]["text"] == _TEXT


def test_english_upsert_call_is_unchanged():
    index = _FakeIndex()
    upsert_embeddings(_embedded(), index)
    vectors, kwargs = index.calls[0]
    assert kwargs == {}
    assert vectors[0]["metadata"]["language"] == "en"
