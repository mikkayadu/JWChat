import pytest
from fastapi.testclient import TestClient

import languages
from api import app as app_module
from api.services import scripture_service as ss

_PAGE = """
<html><body>
<span class="verse" id="v43003016"><sup class="verseNum">16</sup>Na Onyankopɔn dɔ wiase paa
<a class="xrefLink">+</a>, ma enti ɔde ne Ba a ɔwoo no koro no mae<a class="footnoteLink">*</a>.</span>
<span class="verse" id="v43003017"><sup class="verseNum">17</sup>Onyankopɔn ansoma ne Ba no.</span>
</body></html>
"""


@pytest.fixture(autouse=True)
def _twi_enabled(monkeypatch):
    monkeypatch.setenv("ENABLED_LANGUAGES", "en,tw")


def test_verse_anchor_is_unpadded_for_low_book_numbers():
    assert ss.verse_anchor(1, 1, 1) == "v1001001"
    assert ss.verse_anchor(43, 3, 16) == "v43003016"


def test_finder_url():
    assert ss.finder_url(languages.get("tw"), 43, 3) == (
        "https://www.jw.org/finder?wtlocale=TW&pub=nwt&bible=43003001"
    )
    assert ss.finder_url(languages.get("en"), 1, 2, 3) == (
        "https://www.jw.org/finder?wtlocale=E&pub=nwtsty&bible=1002003"
    )


def test_get_verses_matches_by_id(monkeypatch):
    url = "https://www.jw.org/tw/nhomakorabea/bible/nwt/nhoma/yohane/3/"
    monkeypatch.setattr(ss, "_fetch_chapter", lambda lang, book, chapter: (url, _PAGE))
    got_url, verses = ss.get_verses("tw", 43, 3, 16, 17)
    assert got_url == url
    assert verses == [
        {"verse": 16, "text": "Na Onyankopɔn dɔ wiase paa, ma enti ɔde ne Ba a ɔwoo no koro no mae."},
        {"verse": 17, "text": "Onyankopɔn ansoma ne Ba no."},
    ]


def test_get_verses_raises_when_missing(monkeypatch):
    monkeypatch.setattr(ss, "_fetch_chapter", lambda *_: ("u", _PAGE))
    with pytest.raises(ss.ScriptureNotFound):
        ss.get_verses("tw", 43, 3, 40)


def test_scripture_endpoint_accepts_lang_and_ignores_slug(monkeypatch):
    calls = []

    def fake_get_verses(lang, book, chapter, verse, end_verse):
        calls.append((lang, book, chapter, verse, end_verse))
        return "https://jw.org/x/", [{"verse": verse, "text": "t"}]

    monkeypatch.setattr(app_module, "get_verses", fake_get_verses)
    client = TestClient(app_module.app)

    resp = client.get("/api/scripture", params={"book": 43, "chapter": 3, "verse": 16, "lang": "tw"})
    assert resp.status_code == 200
    assert resp.json() == {"reference": "3:16", "url": "https://jw.org/x/", "verses": [{"verse": 16, "text": "t"}]}

    resp = client.get("/api/scripture", params={"slug": "john", "book": 43, "chapter": 3, "verse": 16})
    assert resp.status_code == 200
    assert calls == [("tw", 43, 3, 16, None), ("en", 43, 3, 16, None)]


def test_scripture_endpoint_rejects_disabled_language(monkeypatch):
    monkeypatch.setenv("ENABLED_LANGUAGES", "en")
    client = TestClient(app_module.app)
    resp = client.get("/api/scripture", params={"book": 43, "chapter": 3, "verse": 16, "lang": "tw"})
    assert resp.status_code == 400


def test_languages_endpoint():
    client = TestClient(app_module.app)
    body = client.get("/api/languages").json()
    assert [item["code"] for item in body] == ["en", "tw"]
    assert body[1]["native_name"] == "Twi"
    assert body[1]["book_names"][42] == "Yohane"
    assert len(body[0]["book_names"]) == 66
