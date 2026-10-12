import pytest


@pytest.fixture(autouse=True)
def ingested_urls_file(tmp_path, monkeypatch):
    """Every upsert records its source; keep tests out of the real ingested_urls.txt."""
    path = tmp_path / "ingested_urls.txt"
    monkeypatch.setenv("INGESTED_URLS_FILE", str(path))
    return path
