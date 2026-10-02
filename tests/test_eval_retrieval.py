import json

import pytest

from scripts.eval_retrieval import first_hit_rank, load_questions, normalise_url, percentile, summarise


def test_normalise_url_ignores_scheme_slash_and_fragment():
    assert normalise_url("https://www.jw.org/tw/x/") == normalise_url("http://WWW.jw.org/tw/x#top")


def test_first_hit_rank():
    retrieved = ["https://www.jw.org/tw/a/", "https://www.jw.org/tw/b/", "https://www.jw.org/tw/c/"]
    assert first_hit_rank(retrieved, ["https://www.jw.org/tw/b"]) == 2
    assert first_hit_rank(retrieved, ["https://www.jw.org/tw/z"]) is None


def test_summarise_hit_and_mrr():
    metrics = summarise([1, 2, None, 7], k=5)
    assert metrics["hit@5"] == pytest.approx(0.5)
    assert metrics["mrr"] == pytest.approx((1 + 0.5) / 4)


def test_percentile():
    assert percentile([0.1, 0.2, 0.3, 0.4, 0.5], 50) == pytest.approx(0.3)
    assert percentile([0.1, 0.5], 25) == pytest.approx(0.2)
    assert percentile([], 50) == 0.0


def test_load_questions_rejects_missing_fields(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps({"question": "Hwan ne Yehowa?"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_questions(path)
