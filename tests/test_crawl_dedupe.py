from ingestion.tasks import worker

_NAV = "https://www.jw.org/en/news"
_ARTICLE = "https://www.jw.org/en/bible-teachings/questions/x"


class _FakeRedis:
    def __init__(self):
        self.sets, self.counters = {}, {}

    def sadd(self, key, value):
        members = self.sets.setdefault(key, set())
        is_new = value not in members
        members.add(value)
        return int(is_new)

    def expire(self, *_):
        pass

    def get(self, key):
        return self.counters.get(key)

    def incr(self, key):
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]


def test_links_shared_between_pages_are_queued_once(monkeypatch):
    queued, redis_client = [], _FakeRedis()
    monkeypatch.setattr(worker, "_get_redis_client", lambda: redis_client)
    monkeypatch.setattr("ingestion.extractors.crawler._fetch", lambda url: "<html></html>")
    monkeypatch.setattr(
        "ingestion.extractors.crawler._extract_links",
        lambda *_, **__: [_NAV, _NAV + "/", _ARTICLE],
    )
    monkeypatch.setattr(worker.ingest_url, "delay", lambda *_, **__: None)
    monkeypatch.setattr(worker.crawl_and_enqueue, "delay", lambda link, *_, **__: queued.append(link))

    worker.crawl_and_enqueue("https://www.jw.org/en/a", "job", "www.jw.org", respect_robots=False)
    worker.crawl_and_enqueue("https://www.jw.org/en/b", "job", "www.jw.org", respect_robots=False)

    assert queued == [_NAV, _ARTICLE]
