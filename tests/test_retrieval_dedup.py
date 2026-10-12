from api.services.chat_service import _content_key

BODY = "Would you welcome a visit? Even in this troubled world, you can gain happiness."


def test_reprint_under_another_title_is_a_duplicate():
    a = f"[Source: Would You Welcome a Visit?]\n{BODY}"
    b = f"[Source: The Watchtower, January 2008]\n{BODY}"
    assert _content_key(a) == _content_key(b)


def test_passages_under_one_long_header_stay_apart():
    header = "[Source: Stay Faithful Through the \"Great Tribulation\" | Watchtower Study | Section: What Lies Ahead?]\n"
    assert _content_key(header + "First passage.") != _content_key(header + "Second passage.")


def test_whitespace_and_case_are_ignored():
    assert _content_key(BODY) == _content_key("  " + BODY.upper().replace(" ", "\n"))
