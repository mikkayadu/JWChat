import pytest

import languages


def test_registry_has_66_books_per_language():
    for lang in languages.LANGUAGES.values():
        assert len(lang.book_names) == 66, lang.code


def test_english_keeps_default_namespace():
    assert languages.get("en").namespace == ""
    assert languages.get("tw").namespace == "tw"


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://www.jw.org/tw/nhomakorabea/", "tw"),
        ("https://www.jw.org/en/library/books/", "en"),
        ("https://jw.org/tw/", "tw"),
        ("https://www.jw.org/fr/bibliotheque/", None),  # unregistered locale
        ("https://www.jw.org/", None),
        ("https://example.com/tw/page", None),
        ("https://notjw.org/tw/page", None),
    ],
)
def test_from_jw_url(url, expected):
    lang = languages.from_jw_url(url)
    assert (lang.code if lang else None) == expected


def test_resolve_ingest_language_order():
    twi_url = "https://www.jw.org/tw/nhomakorabea/"
    assert languages.resolve_ingest_language(twi_url).code == "tw"
    assert languages.resolve_ingest_language(twi_url, "en").code == "en"
    assert languages.resolve_ingest_language("./docs/file.pdf").code == "en"
    assert languages.resolve_ingest_language("./docs/file.pdf", "tw").code == "tw"


def test_enabled_defaults_to_english(monkeypatch):
    monkeypatch.delenv("ENABLED_LANGUAGES", raising=False)
    assert [lang.code for lang in languages.enabled()] == ["en"]


def test_enabled_always_includes_english(monkeypatch):
    monkeypatch.setenv("ENABLED_LANGUAGES", "tw, xx")
    assert [lang.code for lang in languages.enabled()] == ["en", "tw"]
