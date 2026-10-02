from ingestion.extractors.html_scraper import _parse_html

_BODY = "<p>" + " ".join(["Na Onyankopɔn dɔ wiase paa."] * 40) + "</p>"
_HTML = f"<html><head><title>Asɛm</title></head><body><article>{_BODY}</article></body></html>"


def test_language_from_url():
    page = _parse_html(_HTML, "https://www.jw.org/tw/nhomakorabea/x/", status_code=200)
    assert page.metadata["language"] == "tw"


def test_language_override_wins():
    page = _parse_html(_HTML, "https://www.jw.org/tw/nhomakorabea/x/", status_code=200, language="en")
    assert page.metadata["language"] == "en"
