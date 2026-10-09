import unicodedata

from ingestion.utils.embedder import _is_low_quality_error_chunk, _normalize_metadata_text


def test_twi_letters_survive():
    text = "Na Onyankopɔn dɔ wiase paa. Ɛno nti"
    assert _normalize_metadata_text(text) == text


def test_non_latin1_scripts_survive():
    text = "agápē (ἀγάπη) and ḥesed (חֶסֶד)"
    assert _normalize_metadata_text(text) == unicodedata.normalize("NFC", text)


def test_output_is_nfc():
    decomposed = "Onyankopón"  # o + combining acute
    assert _normalize_metadata_text(decomposed) == "Onyankopón"


def test_quotes_and_dashes_are_folded():
    assert _normalize_metadata_text("“Yes” — ‘no’ – ok") == "\"Yes\" - 'no' - ok"


def test_zero_width_characters_are_removed():
    assert _normalize_metadata_text("a​b﻿c") == "abc"


def test_empty_input():
    assert _normalize_metadata_text("") == ""
    assert _normalize_metadata_text(None) == ""


def test_article_words_do_not_mark_a_chunk_as_an_error_page():
    text = "Eve ate the forbidden fruit, and the first human couple turned away from God and suffered."
    assert not _is_low_quality_error_chunk(text)
