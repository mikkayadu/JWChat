from ingestion.extractors.crawler import _extract_links, _in_path_scope, _is_excluded_url

PAGE = """
<html><body>
  <a href="/tw/nhomakorabea/nhoma/">Twi library</a>
  <a href="https://www.jw.org/tw/nsɛm-foforo/">Twi news</a>
  <a href="/en/library/">English</a>
  <a href="/fr/bibliotheque/">French</a>
  <a href="/tw/nhomakorabea/videos/#tw/mediaitems/VODBible/pub-a_VIDEO">Twi video</a>
  <a href="/tw/nhomakorabea/videos/#en/mediaitems/VODBible/pub-a_VIDEO">English video</a>
  <a href="https://other.org/tw/page">Off-site</a>
  <a href="#top">Anchor</a>
</body></html>
"""


def test_extract_links_keeps_only_prefix():
    links = _extract_links(PAGE, "https://www.jw.org/tw/", "www.jw.org", False, "/tw/")
    assert "https://www.jw.org/tw/nhomakorabea/nhoma" in links
    assert any("/tw/nsɛm-foforo" in link for link in links)
    assert "https://www.jw.org/tw/nhomakorabea/videos/#tw/mediaitems/VODBible/pub-a_VIDEO" in links
    assert not any("/en/" in link or "/fr/" in link for link in links)
    assert not any("#en/" in link for link in links)
    assert not any("other.org" in link for link in links)


def test_extract_links_without_prefix_is_unchanged():
    links = _extract_links(PAGE, "https://www.jw.org/tw/", "www.jw.org", False)
    assert "https://www.jw.org/en/library" in links
    assert "https://www.jw.org/fr/bibliotheque" in links


def test_in_path_scope_accepts_normalised_seed():
    assert _in_path_scope("/tw", "/tw/")
    assert _in_path_scope("/tw/nhomakorabea", "/tw/")
    assert not _in_path_scope("/twi", "/tw/")
    assert _in_path_scope("/anything", None)


def test_bible_pages_excluded_in_both_languages():
    assert _is_excluded_url("https://www.jw.org/en/library/bible/study-bible/books/john/3/")
    assert _is_excluded_url("https://www.jw.org/tw/nhomakorabea/bible/nwt/nhoma/yohane/3/")
    assert not _is_excluded_url("https://www.jw.org/tw/nhomakorabea/nhoma/")


def test_language_picker_and_redirect_pages_excluded():
    assert _is_excluded_url("https://www.jw.org/en/choose-language?docid=1011756&locale=en&prefer=lang")
    assert _is_excluded_url("https://www.jw.org/tw/choose-language?docid=1011756")
    assert _is_excluded_url("https://www.jw.org/open?lank=pub-cpw_x_VIDEO&wtlocale=E")
    assert not _is_excluded_url("https://www.jw.org/en/bible-teachings/questions/open-minded/")
