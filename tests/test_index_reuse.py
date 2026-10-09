from ingestion.utils import embedder
from ingestion.utils.embedder import init_collection


def test_index_handle_is_reused(monkeypatch):
    listed = []

    class FakePinecone:
        def __init__(self, api_key):
            pass

        def list_indexes(self):
            listed.append(True)
            return [{"name": "jwchat"}]

        def Index(self, name):
            return f"index:{name}"

    monkeypatch.setattr("pinecone.Pinecone", FakePinecone)
    monkeypatch.setattr(embedder, "_indexes", {})
    monkeypatch.setenv("PINECONE_API_KEY", "k")
    monkeypatch.setenv("PINECONE_INDEX", "jwchat")
    assert init_collection() == init_collection() == "index:jwchat"
    assert len(listed) == 1
