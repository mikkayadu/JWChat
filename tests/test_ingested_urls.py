from ingestion.cli.pipeline import _read_lines
from ingestion.utils.chunker import TextChunk
from ingestion.utils.embedder import upsert_embeddings
from ingestion.utils.ingested_urls import _HEADER, read_ingested_urls, record_ingested, write_ingested_urls

_A = "https://www.jw.org/en/bible-teachings/questions/original-sin"
_B = "https://www.jw.org/en/bible-teachings/questions/alpha-omega"
_C = "https://www.jw.org/tw/nhomakorabea/x"
_TEXT = "God loved the world so much that he gave his only-begotten Son so that everyone exercising faith"


class _FakeIndex:
    def upsert(self, vectors, **kwargs):
        pass


def test_sources_are_appended_and_read_back_once(ingested_urls_file):
    write_ingested_urls([_A])
    record_ingested([_B, _B])
    record_ingested([_A, _C])
    assert ingested_urls_file.read_text() == f"{_HEADER}{_A}\n{_B}\n{_A}\n{_C}\n"
    assert read_ingested_urls() == [_A, _B, _C]


def test_upsert_records_the_source():
    chunks = [TextChunk(text=_TEXT, chunk_index=i, token_count=20, source=_A) for i in range(2)]
    upsert_embeddings([{"chunk": c, "vector": [0.1, 0.2]} for c in chunks], _FakeIndex())
    assert read_ingested_urls() == [_A]


def test_list_works_as_urls_input(ingested_urls_file):
    write_ingested_urls([_B, _A, _B])
    assert ingested_urls_file.read_text().startswith(_HEADER)
    assert _read_lines(str(ingested_urls_file)) == [_B, _A]


def test_unwritable_list_does_not_fail_ingestion(ingested_urls_file):
    ingested_urls_file.mkdir()  # a directory where the file should be
    record_ingested([_A])  # logs a warning instead of raising
