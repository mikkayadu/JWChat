import asyncio
from types import SimpleNamespace

import pytest

import languages
from api.services.chat_service import ChatService
from api.services.ensemble import RAGEnsemble
from api.services.ensemble.strategies.base import (
    FINAL_ANSWER_PROMPT,
    ReasoningStrategy,
    language_rules,
)
from api.services.ensemble.strategies.cot import _COT_PROMPT

TWI_Q = "Dɛn nti na Onyankopɔn ma amanehunu kɔ so?"
TWI_Q_EN = "Why does God allow suffering to continue?"
EN_Q = "Why does God allow suffering?"


@pytest.fixture(autouse=True)
def _twi_enabled(monkeypatch):
    monkeypatch.setenv("ENABLED_LANGUAGES", "en,tw")
    monkeypatch.setenv("MAX_CHUNKS_PER_SOURCE", "0")


class FakeEmbeddings:
    """Embeds text as [text] so the fake index can see which text was searched."""

    def embed_query(self, text):
        return [text]


class FakeIndex:
    def __init__(self, hits_by_namespace):
        self.hits = hits_by_namespace
        self.calls = []

    def query(self, vector, top_k, include_metadata, **kwargs):
        namespace = kwargs.get("namespace", "")
        self.calls.append({"namespace": namespace, "text": vector[0], "kwargs": kwargs})
        matches = [
            {"score": 0.9 - i * 0.01, "metadata": {"text": f"{namespace or 'en'} passage {i} " + "x " * 5,
                                                    "source": f"https://www.jw.org/{namespace or 'en'}/{i}"}}
            for i in range(self.hits.get(namespace, 0))
        ]
        return {"matches": matches}


class FakeIntent:
    def __init__(self, detected="en", english_query="", greeting=0.0, ood=0.0, fail=False):
        self.result = SimpleNamespace(
            greeting_score=greeting,
            out_of_domain_score=ood,
            detected_language=detected,
            english_query=english_query,
        )
        self.fail = fail

    def invoke(self, _):
        if self.fail:
            raise RuntimeError("model down")
        return self.result


class FakeChain:
    def __init__(self, label):
        self.label = label
        self.inputs = []

    def invoke(self, inputs):
        self.inputs.append(inputs)
        return SimpleNamespace(content=f"{self.label} answer")

    def stream(self, inputs):
        self.inputs.append(inputs)
        yield SimpleNamespace(content=f"{self.label} answer")


class FakeReranker:
    def __init__(self):
        self.calls = []

    def compress_documents(self, documents, query):
        self.calls.append((query, [d.metadata["language"] for d in documents]))
        for d in documents:
            d.metadata["relevance_score"] = 0.5
        return documents


def make_service(hits, intent=None, reranker=None):
    svc = ChatService.__new__(ChatService)
    svc._embeddings = FakeEmbeddings()
    svc._index = FakeIndex(hits)
    svc._reranker = reranker
    svc._multi_query_chain = None
    svc._hyde_chain = None
    svc._use_hyde = False
    svc._retrieval_k = 20
    svc._rerank_top_n = 5
    svc._bm25_alpha = 0.6
    svc._fallback_min_hits = 2
    svc._bm25_fold = languages.bm25_fold_table()
    svc._greeting_threshold = 0.75
    svc._out_of_domain_threshold = 0.65
    svc._intent_chain = intent or FakeIntent()
    svc._answer_chains = {"en": FakeChain("EN"), "tw": FakeChain("TW")}
    svc._answer_chain = svc._answer_chains["en"]
    svc._ensemble = None
    svc._ensemble_show_metadata = False
    svc._graph = svc._build_graph()
    return svc


# ── language resolution ──────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "requested, detected, expected",
    [
        ("auto", "tw", "tw"),
        ("auto", "en", "en"),
        ("auto", "ak", "tw"),       # alias a detector might return
        ("auto", "en-US", "en"),
        ("auto", "fr", "en"),       # detected but not enabled
        ("auto", "", "en"),         # detection failed
        ("en", "tw", "en"),         # picker wins
        ("tw", "en", "tw"),
        ("tw", "", "tw"),           # classifier outage keeps the requested language
    ],
)
def test_answer_language_resolution(requested, detected, expected):
    fields = ChatService._language_fields(requested, detected, "", "q")
    assert fields["answer_language"] == expected


def test_disabled_language_resolves_to_english(monkeypatch):
    monkeypatch.setenv("ENABLED_LANGUAGES", "en")
    fields = ChatService._language_fields("auto", "tw", TWI_Q_EN, TWI_Q)
    assert fields["detected_language"] == "tw"
    assert fields["answer_language"] == "en"
    assert fields["english_query"] == TWI_Q_EN


def test_empty_english_query_falls_back_to_question():
    assert ChatService._language_fields("auto", "en", "", EN_Q)["english_query"] == EN_Q


# ── retrieval query rule ─────────────────────────────────────────────────────

def _retrieve(svc, question, requested="auto"):
    state = {"question": question, "top_k": 5, "mode": "normal", "requested_language": requested}
    state.update(svc._classify_intent_node(state))
    state.update(svc._retrieve_node(state))
    return state


def test_twi_question_with_enough_twi_hits_skips_fallback():
    svc = make_service({"tw": 3, "": 5}, FakeIntent("tw", TWI_Q_EN))
    state = _retrieve(svc, TWI_Q)
    assert svc._index.calls == [{"namespace": "tw", "text": TWI_Q, "kwargs": {"namespace": "tw"}}]
    assert state["answer_language"] == "tw"
    assert state["used_fallback"] is False


def test_thin_twi_hits_fall_back_to_english_with_translation():
    svc = make_service({"tw": 1, "": 5}, FakeIntent("tw", TWI_Q_EN))
    state = _retrieve(svc, TWI_Q)
    assert [(c["namespace"], c["text"]) for c in svc._index.calls] == [("tw", TWI_Q), ("", TWI_Q_EN)]
    assert svc._index.calls[1]["kwargs"] == {}  # English stays on the default namespace
    assert state["used_fallback"] is True
    assert [g["language"] for g in state["retrieval_groups"]] == ["tw", "en"]
    assert state["retrieved"][0][0].metadata["language"] == "tw"  # Twi group first


def test_fallback_with_no_english_hits_is_not_used():
    svc = make_service({"tw": 0, "": 0}, FakeIntent("tw", TWI_Q_EN))
    state = _retrieve(svc, TWI_Q)
    assert len(svc._index.calls) == 2
    assert state["used_fallback"] is False


def test_picker_english_with_twi_question_searches_english_with_translation():
    svc = make_service({"tw": 5, "": 5}, FakeIntent("tw", TWI_Q_EN))
    state = _retrieve(svc, TWI_Q, requested="en")
    assert [(c["namespace"], c["text"]) for c in svc._index.calls] == [("", TWI_Q_EN)]
    assert state["answer_language"] == "en"


def test_english_question_is_unchanged():
    svc = make_service({"": 5}, FakeIntent("en", ""))
    state = _retrieve(svc, EN_Q)
    assert svc._index.calls == [{"namespace": "", "text": EN_Q, "kwargs": {}}]
    assert state["answer_language"] == "en"
    assert state["used_fallback"] is False


def test_each_group_is_reranked_with_its_own_query():
    reranker = FakeReranker()
    svc = make_service({"tw": 1, "": 2}, FakeIntent("tw", TWI_Q_EN), reranker=reranker)
    state = {"question": TWI_Q, "top_k": 5, "mode": "thinking", "requested_language": "auto"}
    state.update(svc._classify_intent_node(state))
    state.update(svc._retrieve_node(state))
    state.update(svc._rerank_node(state))
    assert reranker.calls == [(TWI_Q, ["tw"]), (TWI_Q_EN, ["en", "en"])]
    assert [d.metadata["language"] for d, _ in state["reranked_docs"]] == ["tw", "en", "en"]


# ── BM25 folding ─────────────────────────────────────────────────────────────

def test_bm25_folds_twi_letters():
    svc = make_service({})
    assert svc._bm25_tokens("Onyankopɔn Ɛno") == svc._bm25_tokens("Onyankopon Eno") == ["onyankopon", "eno"]


# ── context and prompts ──────────────────────────────────────────────────────

def test_context_labels_languages_only_for_non_english():
    svc = make_service({"tw": 1, "": 2}, FakeIntent("tw", TWI_Q_EN))
    state = _retrieve(svc, TWI_Q)
    context = svc._build_context_node(state)["context"]
    assert "(Language: Twi)" in context and "(Language: English)" in context

    svc = make_service({"": 2}, FakeIntent("en"))
    state = _retrieve(svc, EN_Q)
    assert "(Language:" not in svc._build_context_node(state)["context"]


def test_language_rules():
    assert language_rules("en") == ""
    rules = language_rules("tw")
    tw = languages.get("tw")
    assert "Write the whole answer in Twi" in rules
    assert tw.messages.english_sources_note in rules
    assert tw.messages.not_found in rules
    assert "Yohane 3:16" in rules
    assert "Adiyisɛm" in rules


def test_prompts_render_with_language_rules():
    for prompt, extra in (
        (FINAL_ANSWER_PROMPT, {"notes_block": ""}),
        (_COT_PROMPT, {}),
    ):
        english = prompt.format(question="q", context="c", language_rules=language_rules("en"), **extra)
        twi = prompt.format(question="q", context="c", language_rules=language_rules("tw"), **extra)
        assert "LANGUAGE RULES" not in english
        assert "LANGUAGE RULES" in twi


def test_answer_chain_gets_language_rules():
    svc = make_service({"tw": 3}, FakeIntent("tw", TWI_Q_EN))
    result = svc.ask(TWI_Q, language="auto")
    assert result["answer"] == "TW answer"
    assert result["answer_language"] == "tw"
    assert "LANGUAGE RULES" in svc._answer_chains["tw"].inputs[0]["language_rules"]


def test_english_answer_has_no_language_rules():
    svc = make_service({"": 3}, FakeIntent("en"))
    result = svc.ask(EN_Q)
    assert result["answer"] == "EN answer"
    assert svc._answer_chains["en"].inputs[0]["language_rules"] == ""


# ── canned replies and stream events ─────────────────────────────────────────

def test_twi_greeting_is_canned_in_twi():
    svc = make_service({}, FakeIntent("tw", "Good morning", greeting=0.95))
    assert svc.ask("Maakye")["answer"] == languages.get("tw").messages.greeting


def test_classifier_outage_uses_requested_language():
    svc = make_service({}, FakeIntent(fail=True))
    assert svc.ask(TWI_Q, language="tw")["answer"] == languages.get("tw").messages.unavailable
    assert svc.ask(TWI_Q, language="auto")["answer"] == languages.get("en").messages.unavailable


def test_no_results_message_in_answer_language():
    svc = make_service({"tw": 0, "": 0}, FakeIntent("tw", TWI_Q_EN))
    assert svc.ask(TWI_Q)["answer"] == languages.get("tw").messages.no_results


def test_stream_emits_meta_only_when_asked():
    svc = make_service({"tw": 1, "": 2}, FakeIntent("tw", TWI_Q_EN))
    events = list(svc.ask_stream(TWI_Q, language="auto", emit_meta=True))
    assert events[0] == {"event": "meta", "data": {"answer_language": "tw", "used_fallback": True}}
    assert events[1]["event"] == "sources"

    svc = make_service({"": 2}, FakeIntent("en"))
    events = list(svc.ask_stream(EN_Q))
    assert [e["event"] for e in events] == ["sources", "token", "done"]


def test_stream_canned_reply_emits_meta():
    svc = make_service({}, FakeIntent("tw", "Good morning", greeting=0.95))
    events = list(svc.ask_stream("Maakye", emit_meta=True))
    assert [e["event"] for e in events] == ["meta", "sources", "token", "done"]
    assert events[2]["data"]["text"] == languages.get("tw").messages.greeting


# ── ensemble threading ───────────────────────────────────────────────────────

class _RecordingStrategy(ReasoningStrategy):
    name = "recording"

    def __init__(self):
        super().__init__(llm=None)
        self.languages = []

    async def arun(self, question, context, answer_language="en"):
        self.languages.append(answer_language)
        return f"answer in {answer_language}"


def test_ensemble_threads_answer_language():
    a, b = _RecordingStrategy(), _RecordingStrategy()
    b.name = "recording_b"

    class _Embeddings:
        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

    ensemble = RAGEnsemble({"a": a, "b": b}, _Embeddings())
    result = asyncio.run(ensemble.run("q", "c", "tw"))
    assert a.languages == ["tw"] and b.languages == ["tw"]
    assert result["winner"]["answer"] == "answer in tw"


# ── API validation ───────────────────────────────────────────────────────────

def test_chat_request_language_validation(monkeypatch):
    from pydantic import ValidationError

    from api.app import ChatRequest

    assert ChatRequest(question="Hwan ne Yehowa?").language is None
    assert ChatRequest(question="Hwan ne Yehowa?", language="TW").language == "tw"
    assert ChatRequest(question="Hwan ne Yehowa?", language="auto").language == "auto"
    with pytest.raises(ValidationError):
        ChatRequest(question="Hwan ne Yehowa?", language="xx")
    monkeypatch.setenv("ENABLED_LANGUAGES", "en")
    with pytest.raises(ValidationError):
        ChatRequest(question="Hwan ne Yehowa?", language="tw")
