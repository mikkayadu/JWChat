import hashlib
import os
import re
from typing import TypedDict, List, Tuple, Optional
from enum import Enum

import languages
from api.services.ensemble import RAGEnsemble, STRATEGY_MAP, EnsembleResult
from api.services.ensemble.strategies.base import language_rules
from api.services.providers import ModelFactory
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from loguru import logger
from pinecone import Pinecone
from pydantic import BaseModel, Field

# BM25 — import rank_bm25 directly to avoid langchain_community's Pydantic V1
# dependency, which is incompatible with Python 3.14+
ADVANCED_RETRIEVAL_AVAILABLE = False
try:
    from rank_bm25 import BM25Okapi
    ADVANCED_RETRIEVAL_AVAILABLE = True
except ImportError:
    logger.warning("BM25 retrieval unavailable. Run: .venv/Scripts/pip install rank-bm25")

# Pinecone reranking — langchain-pinecone wrapper tried first; raw inference API
# is the fallback (also broken by Pydantic V1 on Python 3.14, handled at runtime)
_LANGCHAIN_RERANK_AVAILABLE = False
try:
    from langchain_pinecone import PineconeRerank
    _LANGCHAIN_RERANK_AVAILABLE = True
except ImportError:
    pass  # _PineconeInferenceReranker (defined below) handles this path

try:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    HYDE_AVAILABLE = True
except ImportError:
    HYDE_AVAILABLE = False
    logger.warning("HyDE query transformation not available.")


# Hosted models like to sprinkle "smart" typography into their prose, and some of
# it silently breaks the frontend's parsers: MessageBubble's scripture regex only
# accepts [-] or [–] in a verse range, so a non-breaking hyphen turns
# "Ecclesiastes 9:5-6" into a link to verse 5 alone, and a zero-width space
# (which JS \s does NOT match) splits a reference in half. Fold only the
# characters that break parsing - curly quotes and em/en dashes are legitimate
# typography and render fine, so they are left alone.
_TEXT_NORMALIZATION = {
    0x200B: None,  # zero-width space
    0x200C: None,  # zero-width non-joiner
    0x200D: None,  # zero-width joiner
    0xFEFF: None,  # zero-width no-break space (BOM)
    0x00A0: " ",   # no-break space
    0x202F: " ",   # narrow no-break space
    0x2009: " ",   # thin space
    0x2011: "-",   # non-breaking hyphen
}

# The "[Source: title | Section: heading]" line ingestion puts on HTML chunks.
_CONTEXT_HEADER_RE = re.compile(r"\A\[Source:[^\n]*\]\n")


def _content_key(text: str) -> str:
    """Identity of a chunk's text for retrieval dedup: the whole chunk without
    its context header, whitespace collapsed and lowercased. The same article
    reprinted under several URLs collapses to one key, while two passages from
    one section stay apart even though their headers match."""
    body = _CONTEXT_HEADER_RE.sub("", text, count=1)
    return hashlib.sha1(" ".join(body.lower().split()).encode()).hexdigest()


class SourcePayload(TypedDict):
    source: str
    score: float
    snippet: str


class ChatResult(TypedDict, total=False):
    answer: str
    sources: list[SourcePayload]
    answer_language: str
    used_fallback: bool
    ensemble_metadata: dict


class RetrievalGroup(TypedDict):
    """Passages retrieved from one language's namespace, with the query text
    that retrieved them. BM25 and the reranker score each group against its
    own query, since scoring English passages against a Twi question would
    zero their keyword overlap and force a cross-language rerank."""
    language: str
    query: str
    docs: list[tuple[Document, float]]


class ChatState(TypedDict, total=False):
    question: str
    top_k: int
    mode: str
    intent: str
    greeting_score: float
    out_of_domain_score: float
    retrieved: list[tuple[Document, float]]
    context: str
    sources: list[SourcePayload]
    answer: str
    reranked_docs: list[tuple[Document, float]]
    query_variants: list[str]
    hyde_doc: str
    ensemble_metadata: dict | None
    requested_language: str
    detected_language: str
    answer_language: str
    english_query: str
    used_fallback: bool
    retrieval_groups: list[RetrievalGroup]


class _PineconeInferenceReranker:
    """Reranker using the Pinecone inference API directly.

    Drop-in fallback for PineconeRerank when langchain-pinecone's wrapper is
    unavailable (e.g. Python 3.14 / Pydantic V1 incompatibility).
    Exposes the same compress_documents interface so _rerank_node needs no changes.
    """

    def __init__(self, client: Pinecone, model: str, top_n: int) -> None:
        self._client = client
        self._model = model
        self._top_n = top_n

    def compress_documents(self, documents: list[Document], query: str) -> list[Document]:
        results = self._client.inference.rerank(
            model=self._model,
            query=query,
            documents=[doc.page_content for doc in documents],
            top_n=self._top_n,
            return_documents=True,
        )
        reranked: list[Document] = []
        for item in results.data:
            doc = documents[item.index]
            doc.metadata["relevance_score"] = float(item.score)
            reranked.append(doc)
        return reranked


class IntentScores(BaseModel):
    greeting_score: float = Field(ge=0.0, le=1.0)
    out_of_domain_score: float = Field(ge=0.0, le=1.0)
    # Language detection and translation ride on the classifier call that
    # already runs for every request, so the English fallback costs no extra
    # LLM round trip. Defaults and None keep a model that omits them, or
    # returns null for them, from failing validation.
    detected_language: str | None = Field(
        default="",
        description="ISO 639-1 code of the question's language, e.g. 'en' or 'tw'.",
    )
    english_query: str | None = Field(
        default="",
        description="The question translated into English; empty if it is already English.",
    )


class ChatService:
    def __init__(self) -> None:
        self._pinecone_key = self._require_env("PINECONE_API_KEY")
        self._index_name = os.getenv("PINECONE_INDEX", "website-rag")
        self._llm_provider = os.getenv("LLM_PROVIDER", "google")
        self._embeddings_provider = os.getenv("EMBEDDINGS_PROVIDER", "google")
        self._embedding_model = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
        # No hardcoded default: each provider supplies its own (Gemini for google,
        # Llama for ollama, …) when CHAT_MODEL is unset, so switching LLM_PROVIDER
        # doesn't drag a Gemini model name onto another backend.
        self._chat_model = os.getenv("CHAT_MODEL")
        self._target_dim = self._read_target_dim()
        self._greeting_threshold = self._read_threshold("INTENT_GREETING_THRESHOLD", 0.75)
        self._out_of_domain_threshold = self._read_threshold("INTENT_OUT_OF_DOMAIN_THRESHOLD", 0.65)

        # Advanced retrieval configuration
        self._use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"
        self._use_reranking = os.getenv("USE_RERANKING", "true").lower() == "true"
        self._use_hyde = os.getenv("USE_HYDE", "false").lower() == "true"
        self._use_multi_query = os.getenv("USE_MULTI_QUERY", "false").lower() == "true"
        self._rerank_top_n = int(os.getenv("RERANK_TOP_N", "5"))
        self._retrieval_k = int(os.getenv("RETRIEVAL_K", "20"))  # Initial retrieval count
        self._bm25_alpha = float(os.getenv("BM25_HYBRID_ALPHA", "0.6"))  # Weight for dense vs BM25

        # Ensemble reasoning configuration
        self._use_ensemble = os.getenv("USE_ENSEMBLE", "false").lower() == "true"
        self._ensemble_show_metadata = os.getenv("ENSEMBLE_SHOW_METADATA", "false").lower() == "true"
        self._ensemble_similarity_threshold = float(os.getenv("ENSEMBLE_SIMILARITY_THRESHOLD", "0.85"))

        # Fall back to English sources when the answer language's own
        # namespace returns fewer than this many passages above its min score.
        self._fallback_min_hits = int(os.getenv("FALLBACK_MIN_HITS", "2"))
        self._bm25_fold = languages.bm25_fold_table()

        # Credentials/endpoints are resolved by each provider from the environment
        # (GEMINI_API_KEY for google, OLLAMA_BASE_URL for ollama, …) so this
        # service stays provider-agnostic — it only chooses *which* backend.
        self._embeddings = ModelFactory.create_embeddings(
            self._embeddings_provider,
            model=self._embedding_model,
            target_dim=self._target_dim,
        )
        self._pinecone_client = Pinecone(api_key=self._pinecone_key)
        self._index = self._pinecone_client.Index(self._index_name)

        self._reranker = self._init_reranker()
        self._hyde_chain = self._init_hyde() if self._use_hyde else None
        self._multi_query_chain = self._init_multi_query()

        self._answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
"You are a warm, knowledgeable tutor explaining Jehovah's Witness teachings, beliefs, "
"and publications. Your knowledge comes exclusively from the retrieved source passages "
"provided to you in each query — official books, magazines, and jw.org content. You do "
"not draw on outside knowledge, personal opinion, or inference beyond what the sources "
"state.\n\n"

"YOUR ROLE — TEACH, DON'T LIST:\n"
"Imagine you are sitting with one person who is curious and wants to genuinely understand "
"the topic, not just receive a stack of facts. Before you respond, read the retrieved "
"passages carefully and work out what they actually mean and how the ideas connect. Then "
"explain it the way a patient teacher would: start with the core idea, build it out so it "
"makes sense, and connect related points so the reader sees the bigger picture rather than "
"a disconnected list of statements.\n\n"

"GROUNDING RULES (these are absolute):\n"
"1. Every factual claim must come from the retrieved context. Never add claims, conclusions, "
"implications, predictions, or examples that are not present in the sources.\n"
"2. 'Explaining in your own words' means rephrasing and clarifying what the source says — "
"making it easier to understand. It does NOT mean extending the source's reasoning, filling "
"gaps with logic, or stating where an argument 'leads.' If the source does not say it, you "
"do not say it.\n"
"3. If the retrieved context does not contain the answer, respond exactly: "
"'I could not find information about this in the provided sources.' Do not partially answer "
"from guesswork.\n"
"4. If the sources only partially address the question, explain what they do cover and clearly "
"note that they do not address the rest — rather than inventing the missing piece.\n\n"

"HOW TO WRITE THE EXPLANATION:\n"
"- Match your answer length to the question. Simple factual questions ('Who was X?', "
"'When did Y happen?') get a direct, concise answer - a few sentences, not an essay. "
"Save longer explanations for questions that genuinely ask 'why' or 'how.'\n"
"- Open with the central point in plain language, so the reader immediately grasps what this "
"is about.\n"
"- Then unfold the supporting details in a way that flows logically, showing how the points "
"relate to each other (cause, contrast, example, sequence) — but only using relationships the "
"source itself draws.\n"
"- Define any specialized terms the source uses, in plain words.\n"
"- Add the necessary scripture needed in the explanation, but do not quote scripture that is not in the retrieved context.\n"
"- Use a natural, conversational, respectful tone. You are clarifying a belief system as it "
"is taught in these sources, so present it as the sources present it, without editorializing "
"for or against it.\n"
"- Keep it focused. Be thorough enough to make the idea clear, but do not pad. "
"Do not drift into tangentially related topics just because the sources mention them.\n\n"

"CITATIONS:\n"
"- Place an inline citation ([1], [2], etc.) immediately after each sentence or claim it "
"supports.\n"
"- When several sources support one point, list them together: [1][3].\n"
"- Only cite sources you actually drew the claim from.\n"
"- Do not append a list of the sources at the end of your response.\n"
"{language_rules}"
                ),
                (
                    "human",
                    "Question:\n{question}\n\nContext:\n{context}\n\n"
                    "Return an answer with citations.",
                ),
            ]
        )

        self._intent_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
"You are an intent classifier for a question-answering system whose knowledge base is "
"Jehovah's Witness publications and jw.org content. Your job is to score an incoming user "
"message on two dimensions so the system can decide how to route it.\n\n"

"OUTPUT:\n"
"Return two scores, each a float between 0 and 1, plus two language fields:\n"
"- greeting_score: how much the message is purely a greeting or social pleasantry with no "
"information-seeking intent (e.g. 'hi', 'hello', 'thanks', 'how are you', 'good morning'). "
"A message that greets AND asks a real question is NOT a greeting — score it low.\n"
"- out_of_domain_score: how much the message CANNOT be answered from Jehovah's Witness "
"publications and jw.org content. Low = answerable from those sources; high = not.\n"
"- detected_language: the ISO 639-1 code of the language the message is written in "
"(use 'tw' for Twi/Akan, including Twi typed without the letters ɛ and ɔ). For a message "
"mixing languages, give the main one.\n"
"- english_query: the message translated into natural English, for searching English "
"documents. Keep names of people, places and Bible books recognisable. Leave it empty "
"if the message is already in English.\n\n"

"WHAT IS IN DOMAIN (out_of_domain_score should be LOW):\n"
"The knowledge base covers a wide range of life, moral, and existential questions, not only "
"narrow doctrinal ones. Treat as in-domain any question that these sources actually address, "
"including:\n"
"- Bible teachings, prophecy, scripture meaning, and doctrine.\n"
"- Big existential and 'why' questions about life that the Bible speaks to — e.g. 'Why do "
"humans suffer?', 'Why do we die?', 'What happens after death?', 'Does God care about us?', "
"'What is the meaning of life?', 'Is there hope for the future?'\n"
"- Practical and moral life topics these publications discuss — e.g. family, marriage, "
"raising children, coping with grief, anxiety, loneliness, anger, or hardship, viewed from "
"a Bible-based perspective.\n"
"- Questions about God, Jesus, Jehovah, the Kingdom, resurrection, God's purpose, and "
"related themes.\n"
"Do NOT mark a question out-of-domain just because it is broad or philosophical. If the "
"sources plausibly speak to it, it is in domain.\n\n"

"WHAT IS OUT OF DOMAIN (out_of_domain_score should be HIGH):\n"
"- Questions seeking the doctrine or practices of OTHER religions on their own terms "
"(e.g. 'What do Catholics believe about purgatory?', 'Explain the Five Pillars of Islam').\n"
"- General knowledge, factual trivia, science, history, math, news, or current events with "
"no religious dimension (e.g. 'What is the capital of France?', 'How do I write a Python "
"loop?').\n"
"- Requests for personal/professional advice that the Bible does not directly speak to "
"(e.g. medical diagnoses, legal advice, financial or tax help, tech support).\n"
"- Tasks unrelated to the knowledge base (translating arbitrary text, coding, summarizing "
"arbitrary text).\n\n"

"LANGUAGE:\n"
"Messages may be written in any language, or mix languages (e.g. Twi with English words). "
"Score them on their substance exactly as you would the same message in English. A "
"question asked in Twi or any other language is NOT a translation request. Greetings in "
"other languages (e.g. 'Maakye', 'Akwaaba', 'Wo ho te sɛn?') are greetings.\n\n"

"GUIDANCE:\n"
"- The two scores are independent. A pure greeting has high greeting_score; its "
"out_of_domain_score is not meaningful but should default low.\n"
"- When a religious/existential question is phrased casually or emotionally "
"('I feel like life is pointless, what's the point of it all?'), still treat it as in "
"domain — it maps to questions these sources address.\n"
"- Judge by the substance of what is being asked, not by tone or wording. When genuinely "
"uncertain whether the sources cover something, lean toward in-domain (lower "
"out_of_domain_score) so a real question is not wrongly rejected."
                ),
                (
                    "human",
                    "Question:\n{question}",
                ),
            ]
        )

        primary_llm = ModelFactory.create_llm(
            self._llm_provider,
            model=self._chat_model,
            temperature=0,
        )
        # Optional ordered chain of fallback LLMs (e.g. Gemini → Groq 8b → Groq 70b).
        # LangChain's with_fallbacks takes a LIST tried left-to-right: it catches
        # ANY failure of the one before it (rate limit/quota, timeout, 5xx, network)
        # and advances to the next, and it supports .stream(). So callers never
        # branch on which backend answered.
        self._fallback_descs: list[str] = []
        fallback_llms = self._init_fallback_llms()

        # Ensemble strategies need a concrete chat model; give them the primary.
        # Only build it when explicitly enabled — otherwise skip the startup cost.
        self._ensemble = self._init_ensemble(primary_llm) if self._use_ensemble else None

        # One answer chain per enabled language. A language may name its own
        # fallback chain (FALLBACKS_TW=...) so a weak fallback model that writes
        # poor Twi is never used for Twi; otherwise it shares FALLBACKS.
        self._answer_chains = {}
        for lang in languages.enabled():
            spec_env = f"FALLBACKS_{lang.code.upper()}"
            if lang.code != languages.DEFAULT.code and os.getenv(spec_env) is not None:
                lang_descs: list[str] = []
                lang_fallbacks = self._init_fallback_llms(spec_env, legacy=False, descs=lang_descs)
            else:
                lang_fallbacks = fallback_llms
            answer_llm = primary_llm.with_fallbacks(lang_fallbacks) if lang_fallbacks else primary_llm
            self._answer_chains[lang.code] = self._answer_prompt | answer_llm
        self._answer_chain = self._answer_chains[languages.DEFAULT.code]

        # Intent classification uses its own provider/fallback chain, independent
        # of LLM_PROVIDER — see _init_intent_llm for why.
        self._intent_fallback_descs: list[str] = []
        intent_primary_llm = self._init_intent_llm()
        intent_fallback_llms = self._init_fallback_llms(
            spec_env="INTENT_FALLBACKS",
            legacy=False,
            descs=self._intent_fallback_descs,
        )
        intent_runnable = intent_primary_llm.with_structured_output(IntentScores)
        if intent_fallback_llms:
            intent_runnable = intent_runnable.with_fallbacks(
                [fb.with_structured_output(IntentScores) for fb in intent_fallback_llms]
            )
        self._intent_chain = self._intent_prompt | intent_runnable

        self._graph = self._build_graph()

        fallback_desc = " → ".join(self._fallback_descs) if self._fallback_descs else "none"
        intent_fallback_desc = (
            " → ".join(self._intent_fallback_descs) if self._intent_fallback_descs else "none"
        )
        logger.info(
            f"[CHATSERVICE] ready | llm_provider={self._llm_provider} "
            f"model={self._chat_model or '(provider default)'} "
            f"fallbacks=[{fallback_desc}] "
            f"intent_provider={os.getenv('INTENT_LLM_PROVIDER', 'google')} "
            f"intent_fallbacks=[{intent_fallback_desc}] "
            f"embeddings_provider={self._embeddings_provider} "
            f"hybrid={self._use_hybrid_search} rerank={self._use_reranking} "
            f"hyde={self._use_hyde} multi_query={self._use_multi_query} "
            f"ensemble={self._use_ensemble} "
            f"languages={list(self._answer_chains)}"
        )

    def _init_reranker(self):
        """Try langchain-pinecone PineconeRerank first; fall back to raw Pinecone inference API."""
        if _LANGCHAIN_RERANK_AVAILABLE:
            try:
                return PineconeRerank(model="bge-reranker-v2-m3", top_n=self._rerank_top_n)
            except Exception as e:
                logger.warning(f"PineconeRerank init failed, falling back to inference API: {e}")

        try:
            reranker = _PineconeInferenceReranker(
                client=self._pinecone_client,
                model="bge-reranker-v2-m3",
                top_n=self._rerank_top_n,
            )
            logger.info("Reranking via Pinecone inference API")
            return reranker
        except Exception as e:
            logger.warning(f"Reranking unavailable: {e}")
            return None

    def _init_hyde(self):
        """Initialize HyDE (Hypothetical Document Embeddings) chain."""
        if not HYDE_AVAILABLE:
            return None
        try:
            llm = ModelFactory.create_llm(
                self._llm_provider,
                model=self._chat_model,
                temperature=0.3,
            )
            hyde_prompt = PromptTemplate(
                input_variables=["question"],
                template=(
                    "Please write a detailed passage that could answer the following question. "
                    "Write it in the same language as the question.\n{question}"
                ),
            )
            return LLMChain(llm=llm, prompt=hyde_prompt)
        except Exception as e:
            logger.warning(f"Failed to initialize HyDE: {e}")
            return None

    def _init_multi_query(self):
        """Initialize the LLM used to generate query variants for multi-query expansion."""
        try:
            return ModelFactory.create_llm(
                self._llm_provider,
                model=self._chat_model,
                temperature=0.3,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize multi-query LLM: {e}")
            return None

    def _init_fallback_llms(
        self,
        spec_env: str = "FALLBACKS",
        *,
        legacy: bool = True,
        descs: list[str] | None = None,
    ) -> list[BaseChatModel]:
        """Build an ordered list of fallback chat models, tried in sequence.

        Configured via ``spec_env`` — a comma-separated list of ``provider:model``
        entries, e.g. ``groq:openai/gpt-oss-20b,groq:qwen/qwen3.8-27b``.
        Each entry is split on the FIRST colon only, because model names may
        themselves contain ':' (e.g. ollama ``qwen2.5:1.5b``). A bare ``provider``
        (no colon) uses that provider's default model.

        When ``legacy`` is True and ``spec_env`` is unset, falls back to the
        legacy single ``FALLBACK_LLM_PROVIDER`` / ``FALLBACK_CHAT_MODEL`` pair.
        Entries that fail to construct are skipped (logged) so one bad provider
        can't break startup. Descriptions are appended to ``descs`` (defaults to
        ``self._fallback_descs``) for the startup log line.
        """
        target_descs = descs if descs is not None else self._fallback_descs

        specs: list[tuple[str, str | None]] = []
        raw = os.getenv(spec_env, "").strip()
        if raw:
            for entry in raw.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                provider, _, model = entry.partition(":")  # split on first ':' only
                specs.append((provider.strip(), model.strip() or None))
        elif legacy and os.getenv("FALLBACK_LLM_PROVIDER"):  # legacy single-fallback config
            specs.append((os.getenv("FALLBACK_LLM_PROVIDER"), os.getenv("FALLBACK_CHAT_MODEL")))

        fallbacks: list[BaseChatModel] = []
        for provider, model in specs:
            try:
                fallbacks.append(ModelFactory.create_llm(provider, model=model, temperature=0))
                desc = f"{provider}:{model or '(default)'}"
                target_descs.append(desc)
                logger.info(f"[CHATSERVICE] {spec_env} entry registered | {desc}")
            except Exception as e:
                logger.warning(
                    f"{spec_env} entry init failed (provider={provider} model={model}): {e} — skipping"
                )
        return fallbacks

    def _init_intent_llm(self) -> BaseChatModel:
        """Build the primary LLM for intent classification.

        Deliberately decoupled from ``LLM_PROVIDER`` (the answer-generation
        backend): classification is a single structured-output call that needs
        reliable instruction-following on a nuanced multi-criteria prompt, which
        small/local models (e.g. Ollama) often get wrong — scoring legitimate
        in-domain questions (e.g. "who was Abraham's wife") as out-of-domain.
        Defaults to Google Gemini, which handles this prompt reliably regardless
        of which provider ends up answering the question.
        """
        provider = os.getenv("INTENT_LLM_PROVIDER", "google")
        model = os.getenv("INTENT_CHAT_MODEL")
        try:
            return ModelFactory.create_llm(provider, model=model, temperature=0)
        except Exception as e:
            logger.warning(
                f"Intent classifier provider '{provider}' failed to init: {e} — "
                f"falling back to main LLM_PROVIDER ('{self._llm_provider}')"
            )
            return ModelFactory.create_llm(self._llm_provider, model=self._chat_model, temperature=0)

    def _init_ensemble(self, llm: BaseChatModel) -> RAGEnsemble | None:
        """Parse ENSEMBLE_STRATEGIES, instantiate strategy objects, build the orchestrator."""
        raw = os.getenv("ENSEMBLE_STRATEGIES", "standard,cot,tot")
        names = [name.strip().lower() for name in raw.split(",") if name.strip()]

        strategies: dict[str, "ReasoningStrategy"] = {}
        for name in names:
            cls = STRATEGY_MAP.get(name)
            if cls is None:
                logger.warning(f"Unknown ensemble strategy '{name}' — skipping")
                continue
            canonical = cls.name
            if canonical not in strategies:
                strategies[canonical] = cls(llm)
                logger.info(f"[ENSEMBLE] registered strategy '{canonical}'")

        if len(strategies) < 2:
            logger.warning(
                f"Ensemble requires at least 2 valid strategies, got {len(strategies)} — "
                f"advanced mode will fall back to single-shot"
            )
            return None

        logger.info(f"[ENSEMBLE] initialized with {len(strategies)} strategies: {list(strategies.keys())}")
        return RAGEnsemble(
            strategies=strategies,
            embeddings=self._embeddings,
            similarity_threshold=self._ensemble_similarity_threshold,
        )

    def ask(
        self,
        question: str,
        top_k: int = 5,
        mode: str = "normal",
        language: str = "auto",
    ) -> ChatResult:
        state = self._graph.invoke(
            {"question": question, "top_k": top_k, "mode": mode, "requested_language": language}
        )
        result: ChatResult = {
            "answer": state.get("answer", "I could not generate a response."),
            "sources": state.get("sources", []),
            "answer_language": state.get("answer_language") or languages.DEFAULT.code,
            "used_fallback": bool(state.get("used_fallback")),
        }
        ensemble_metadata = state.get("ensemble_metadata")
        if ensemble_metadata is not None:
            result["ensemble_metadata"] = ensemble_metadata
        return result

    def ask_stream(
        self,
        question: str,
        top_k: int = 5,
        mode: str = "normal",
        language: str = "auto",
        emit_meta: bool = False,
    ):
        """Stream a chat answer as transport-agnostic events.

        Yields dicts of the form ``{"event": <name>, "data": <payload>}`` where
        ``event`` is one of:
          * ``meta``    — {"answer_language": str, "used_fallback": bool}, emitted
                          once before ``sources``, only when ``emit_meta`` is set
                          (so frontends that predate languages never see it)
          * ``sources`` — list[SourcePayload], emitted once before any tokens
          * ``token``   — {"text": str}, one per streamed chunk
          * ``done``    — {}, terminal success marker
          * ``error``   — {"detail": str}, terminal failure marker

        Reuses the existing graph nodes (so retrieval/rerank honour ``mode``), then
        streams the final answer via the fallback-aware answer chain. The ensemble
        path is intentionally not used here — it produces multiple candidates and
        does not stream cleanly; advanced mode degrades to a streamed single shot.
        """
        state: ChatState = {
            "question": question,
            "top_k": top_k,
            "mode": mode,
            "requested_language": language,
        }
        try:
            state.update(self._classify_intent_node(state))
            intent = self._route_intent(state)

            # Greeting / off-topic / error — canned reply, no LLM generation.
            if intent != "in_domain":
                if intent == "greeting":
                    canned = self._greeting_response_node(state)
                elif intent == "error":
                    canned = self._service_unavailable_node(state)
                else:
                    canned = self._out_of_domain_response_node(state)
                yield from self._canned_events(state, canned["answer"], emit_meta)
                return

            # In-domain: mirror the graph edges (retrieve → rerank → build_context).
            state.update(self._retrieve_node(state))
            state.update(self._rerank_node(state))
            state.update(self._build_context_node(state))
            messages = self._messages(state)

            if not state.get("retrieved"):
                logger.warning("[STREAM] retrieved is empty — no docs passed retrieval/filtering")
                yield from self._canned_events(state, messages.no_results, emit_meta)
                return

            context = (state.get("context") or "").strip()
            if not context:
                yield from self._canned_events(state, messages.no_text, emit_meta)
                return

            if emit_meta:
                yield {"event": "meta", "data": self._meta(state)}
            yield {"event": "sources", "data": state.get("sources", [])}

            answer_language = state.get("answer_language") or languages.DEFAULT.code
            chain = self._answer_chains.get(answer_language, self._answer_chain)
            streamed_any = False
            for chunk in chain.stream(self._answer_inputs(question, context, answer_language)):
                text = getattr(chunk, "content", "") or ""
                if text:
                    streamed_any = True
                    yield {"event": "token", "data": {"text": self._normalize_answer_text(text)}}

            if not streamed_any:
                yield {"event": "token", "data": {"text": "I could not generate a response."}}

            yield {"event": "done", "data": {}}

        except Exception as e:
            logger.error(f"[STREAM] failed: {e}")
            yield {"event": "error", "data": {"detail": self._messages(state).unavailable}}

    def _canned_events(self, state: ChatState, text: str, emit_meta: bool):
        """Events for a reply that needs no LLM generation."""
        if emit_meta:
            yield {"event": "meta", "data": self._meta(state)}
        yield {"event": "sources", "data": []}
        yield {"event": "token", "data": {"text": text}}
        yield {"event": "done", "data": {}}

    @staticmethod
    def _meta(state: ChatState) -> dict:
        return {
            "answer_language": state.get("answer_language") or languages.DEFAULT.code,
            "used_fallback": bool(state.get("used_fallback")),
        }

    def _build_graph(self):
        graph = StateGraph(ChatState)
        graph.add_node("classify_intent", self._classify_intent_node)
        graph.add_node("greeting_response", self._greeting_response_node)
        graph.add_node("out_of_domain_response", self._out_of_domain_response_node)
        graph.add_node("service_unavailable", self._service_unavailable_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("rerank", self._rerank_node)
        graph.add_node("build_context", self._build_context_node)
        graph.add_node("answer", self._answer_node)

        graph.set_entry_point("classify_intent")
        graph.add_conditional_edges(
            "classify_intent",
            self._route_intent,
            {
                "greeting": "greeting_response",
                "out_of_domain": "out_of_domain_response",
                "in_domain": "retrieve",
                "error": "service_unavailable",
                "unknown": "out_of_domain_response",
            },
        )

        graph.add_edge("greeting_response", END)
        graph.add_edge("out_of_domain_response", END)
        graph.add_edge("service_unavailable", END)
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "build_context")
        graph.add_edge("build_context", "answer")
        graph.add_edge("answer", END)
        return graph.compile()

    def _classify_intent_node(self, state: ChatState) -> ChatState:
        question = (state.get("question") or "").strip()
        requested = (state.get("requested_language") or "auto").strip().lower()
        if not question:
            return {
                "intent": "greeting",
                "greeting_score": 1.0,
                "out_of_domain_score": 0.0,
                **self._language_fields(requested, "", "", question),
            }

        try:
            scores = self._intent_chain.invoke({"question": question})
            greeting_score = float(scores.greeting_score)
            out_of_domain_score = float(scores.out_of_domain_score)
            lang_fields = self._language_fields(
                requested, scores.detected_language, scores.english_query, question
            )
            logger.info(
                f"[INTENT] q='{question[:60]}' greeting={greeting_score:.2f} "
                f"out_of_domain={out_of_domain_score:.2f} "
                f"(thresholds greeting>={self._greeting_threshold} ood>={self._out_of_domain_threshold}) "
                f"language requested={requested} detected={lang_fields['detected_language'] or '?'} "
                f"answer={lang_fields['answer_language']}"
            )
        except Exception as e:
            # The classifier LLM (and its fallback) failed — this is a service
            # outage, NOT an off-topic question. Route to a clear "unavailable"
            # message instead of the misleading out-of-domain reply.
            logger.warning(f"Intent classification failed (model/service unavailable): {e}")
            return {
                "intent": "error",
                "greeting_score": 0.0,
                "out_of_domain_score": 0.0,
                **self._language_fields(requested, "", "", question),
            }

        scores_state = {
            "greeting_score": greeting_score,
            "out_of_domain_score": out_of_domain_score,
            **lang_fields,
        }

        if greeting_score >= self._greeting_threshold:
            return {"intent": "greeting", **scores_state}

        if out_of_domain_score >= self._out_of_domain_threshold:
            return {"intent": "out_of_domain", **scores_state}

        return {"intent": "in_domain", **scores_state}

    @staticmethod
    def _language_fields(
        requested: str, detected_raw: str | None, english_query: str | None, question: str
    ) -> ChatState:
        """Resolve the answer language from the picker and the detected language.

        An explicit, enabled picker value always wins. "auto" uses the detected
        language when it is enabled. Everything else (a disabled language, a
        failed detection, a classifier outage) resolves to English. Detection
        isn't limited to enabled languages, so a Twi question asked while Twi
        is off still gets an English translation to search with.
        """
        detected = languages.normalise_code(detected_raw)
        enabled = {lang.code for lang in languages.enabled()}
        if requested and requested != "auto":
            answer = requested if requested in enabled else languages.DEFAULT.code
        else:
            answer = detected if detected in enabled else languages.DEFAULT.code
        return {
            "requested_language": requested,
            "detected_language": detected,
            "answer_language": answer,
            # The classifier leaves this empty for English questions.
            "english_query": (english_query or "").strip() or question,
        }

    @staticmethod
    def _messages(state: ChatState) -> languages.Messages:
        """Canned replies in the request's answer language."""
        code = state.get("answer_language") or languages.DEFAULT.code
        return languages.LANGUAGES.get(code, languages.DEFAULT).messages

    @staticmethod
    def _route_intent(state: ChatState) -> str:
        return str(state.get("intent") or "unknown")

    @staticmethod
    def _greeting_response_node(state: ChatState) -> ChatState:
        return {"answer": ChatService._messages(state).greeting, "sources": []}

    @staticmethod
    def _out_of_domain_response_node(state: ChatState) -> ChatState:
        return {"answer": ChatService._messages(state).out_of_domain, "sources": []}

    @staticmethod
    def _service_unavailable_node(state: ChatState) -> ChatState:
        return {"answer": ChatService._messages(state).unavailable, "sources": []}

    def _retrieve_node(self, state: ChatState) -> ChatState:
        question = (state.get("question") or "").strip()
        if not question:
            return {"retrieved": [], "retrieval_groups": [], "used_fallback": False}

        mode = state.get("mode", "normal")
        top_k = int(state.get("top_k", 5))
        top_k = max(1, min(top_k, 20))
        use_reranking = mode in ("thinking", "advanced") and self._reranker is not None
        use_multi_query = mode in ("thinking", "advanced") and self._multi_query_chain is not None
        fetch_k = self._retrieval_k if use_reranking else top_k * 2
        answer_lang = languages.LANGUAGES.get(
            state.get("answer_language") or languages.DEFAULT.code, languages.DEFAULT
        )

        logger.info(
            f"[RETRIEVE] question='{question}' mode={mode} fetch_k={fetch_k} "
            f"reranking={use_reranking} multi_query={use_multi_query} language={answer_lang.code}"
        )

        # Group 1: the answer language's own namespace, searched with its own text.
        primary, hyde_doc, query_variants = self._search_group(
            answer_lang, self._query_text_for(state, answer_lang), fetch_k, use_multi_query
        )
        groups = [primary]

        # Group 2: English sources when the answer language's own are thin,
        # searched with the English translation of the question.
        used_fallback = False
        if (
            answer_lang.code != languages.DEFAULT.code
            and len(primary["docs"]) < self._fallback_min_hits
        ):
            fallback, _, _ = self._search_group(
                languages.DEFAULT,
                self._query_text_for(state, languages.DEFAULT),
                fetch_k,
                use_multi_query,
            )
            groups.append(fallback)
            used_fallback = bool(fallback["docs"])

        hits = ", ".join(f"{g['language']}={len(g['docs'])}" for g in groups)
        fallback_note = (
            f" english_fallback={'fired' if len(groups) > 1 else 'not needed'}"
            if answer_lang.code != languages.DEFAULT.code
            else ""
        )
        logger.info(f"[RETRIEVE] hits per namespace: {hits}{fallback_note}")

        return {
            "retrieved": [pair for group in groups for pair in group["docs"]],
            "retrieval_groups": groups,
            "used_fallback": used_fallback,
            "hyde_doc": hyde_doc,
            "query_variants": query_variants,
        }

    @staticmethod
    def _query_text_for(state: ChatState, lang: languages.Language) -> str:
        """Text to search ``lang``'s namespace with: the question itself when
        it is written in that language, its English translation for the
        English namespace, else the question as-is."""
        question = (state.get("question") or "").strip()
        if state.get("detected_language") == lang.code:
            return question
        if lang.code == languages.DEFAULT.code and state.get("english_query"):
            return state["english_query"]
        return question

    def _search_group(
        self,
        lang: languages.Language,
        text: str,
        fetch_k: int,
        use_multi_query: bool,
    ) -> tuple[RetrievalGroup, str | None, list[str]]:
        """Retrieve from one language's namespace: query expansion and HyDE,
        fetch, content dedup, per-source cap, then BM25 fusion against ``text``.
        Returns the group plus the HyDE passage and query variants used."""
        queries = [text]
        if use_multi_query:
            try:
                queries = self._expand_queries(text)
                logger.info(f"[MULTI-QUERY] {len(queries)} queries total:")
                for i, q in enumerate(queries):
                    logger.info(f"  [{i}] {q}")
            except Exception as e:
                logger.warning(f"Multi-query expansion failed, using original query: {e}")

        # HyDE — rewrite the primary query as a hypothetical answer passage
        hyde_doc = None
        if self._hyde_chain and self._use_hyde:
            try:
                hyde_doc = self._hyde_chain.invoke({"question": text})["text"]
                queries[0] = hyde_doc
                logger.info(f"[HYDE] hypothetical passage (first 120 chars): {hyde_doc[:120]}")
            except Exception as e:
                logger.warning(f"HyDE failed, using original query: {e}")

        # Fetch for every query variant; deduplicate by content fingerprint
        seen: dict[str, tuple[Document, float]] = {}
        for query in queries:
            hits = self._fetch_from_pinecone(query, fetch_k, lang)
            logger.info(f"[PINECONE] ns='{lang.namespace}' query='{query[:60]}...' → {len(hits)} hits passed filter")
            for doc, score in hits:
                key = _content_key(doc.page_content)
                if key not in seen or score > seen[key][1]:
                    seen[key] = (doc, score)

        retrieved = list(seen.values())
        logger.info(f"[RETRIEVE] {len(retrieved)} unique docs after content dedup")

        # Cap chunks per source so one document doesn't dominate the context
        max_per_source = int(os.getenv("MAX_CHUNKS_PER_SOURCE", "2"))
        if max_per_source > 0:
            source_counts: dict[str, int] = {}
            capped: list[tuple[Document, float]] = []
            for doc, score in sorted(retrieved, key=lambda x: x[1], reverse=True):
                src = str((doc.metadata or {}).get("source") or (doc.metadata or {}).get("source_url") or "unknown")
                source_counts[src] = source_counts.get(src, 0) + 1
                if source_counts[src] <= max_per_source:
                    capped.append((doc, score))
            if len(capped) < len(retrieved):
                logger.info(f"[RETRIEVE] capped {len(retrieved)}→{len(capped)} docs (max {max_per_source}/source)")
            retrieved = capped

        # BM25 hybrid re-scoring — combine dense scores with keyword scores
        if retrieved:
            retrieved = self._apply_bm25_fusion(text, retrieved)
            logger.info(f"[BM25] top-3 fused scores: {[round(s,3) for _,s in retrieved[:5]]}")

        return {"language": lang.code, "query": text, "docs": retrieved}, hyde_doc, queries[1:]

    def _fetch_from_pinecone(
        self, query: str, k: int, lang: languages.Language = languages.DEFAULT
    ) -> list[tuple[Document, float]]:
        """Embed query and fetch top-k matches from ``lang``'s namespace above its min score."""
        query_vector = self._embeddings.embed_query(query)
        result = self._index.query(
            vector=query_vector, top_k=k, include_metadata=True, **lang.namespace_kwargs
        )
        matches = result.get("matches", [])
        min_score = lang.min_score if lang.min_score is not None else self._read_min_score()

        all_scores = [float(m.get("score") or 0.0) for m in matches]
        logger.info(f"[PINECONE RAW] {len(matches)} matches, min_score_filter={min_score}, scores={[round(s,3) for s in all_scores[:10]]}")

        retrieved: list[tuple[Document, float]] = []
        for match in matches:
            metadata = dict(match.get("metadata") or {})
            score = float(match.get("score") or 0.0)
            if score < min_score:
                continue
            # Vectors ingested before languages existed carry no language tag;
            # the namespace they came from says what it is.
            metadata.setdefault("language", lang.code)
            retrieved.append((
                Document(page_content=str(metadata.get("text") or ""), metadata=metadata),
                score,
            ))
        return retrieved

    def _expand_queries(self, question: str) -> list[str]:
        """Ask the LLM for 3 reformulations of the question, return original + variants."""
        prompt = (
            "Generate 3 different reformulations of the following question to improve document retrieval. "
            "Each reformulation should approach the topic from a different angle — "
            "one more specific, one broader, one using different terminology. "
            "Write each reformulation in the same language as the original question. "
            "Return only the reformulations, one per line, no numbering or bullets.\n\n"
            f"Original question: {question}"
        )
        response = self._multi_query_chain.invoke(prompt)
        raw = getattr(response, "content", str(response)).strip()
        variants = [line.strip() for line in raw.splitlines() if line.strip()]
        return [question] + variants[:3]

    def _apply_bm25_fusion(
        self, question: str, retrieved: list[tuple[Document, float]]
    ) -> list[tuple[Document, float]]:
        """Re-score dense results with BM25 and combine via weighted sum (sparse-dense fusion).

        Dense vectors capture semantic similarity; BM25 rewards exact keyword overlap.
        Combining them improves results for both vague and keyword-specific queries.
        alpha controls how much weight the dense score carries.
        """
        if not ADVANCED_RETRIEVAL_AVAILABLE or len(retrieved) < 2:
            return retrieved

        try:
            docs = [doc for doc, _ in retrieved]

            # Tokenise and score directly with BM25Okapi (pure Python, no Pydantic)
            # Fold language-specific letters (ɛ→e, ɔ→o) so a question typed
            # without them still matches passages that use them.
            tokenized_corpus = [self._bm25_tokens(doc.page_content) for doc in docs]
            bm25 = BM25Okapi(tokenized_corpus)
            raw_scores = bm25.get_scores(self._bm25_tokens(question))

            # Normalise to [0, 1] so scores are comparable with dense cosine scores
            max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
            norm_scores = [s / max_score for s in raw_scores]

            fused = [
                (doc, self._bm25_alpha * dense + (1 - self._bm25_alpha) * norm_scores[i])
                for i, (doc, dense) in enumerate(retrieved)
            ]
            return sorted(fused, key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.warning(f"BM25 fusion failed, returning dense results: {e}")
            return retrieved

    def _bm25_tokens(self, text: str) -> list[str]:
        return text.lower().translate(self._bm25_fold).split()

    def _rerank_node(self, state: ChatState) -> ChatState:
        mode = state.get("mode", "normal")
        if mode == "normal" or not self._reranker:
            return state

        groups = state.get("retrieval_groups") or []
        if not groups and state.get("retrieved"):
            groups = [{
                "language": languages.DEFAULT.code,
                "query": state.get("question", ""),
                "docs": state["retrieved"],
            }]
        if not groups:
            return state

        try:
            # Each group is reranked against the text that retrieved it, then
            # the groups are concatenated, answer-language group first.
            reranked: list[tuple[Document, float]] = []
            for group in groups:
                if not group["docs"]:
                    continue
                docs = [doc for doc, score in group["docs"]]
                # PineconeRerank expects documents and query
                reranked_docs = self._reranker.compress_documents(docs, group["query"])
                # Convert back to retrieved format with new scores
                reranked.extend(
                    (doc, doc.metadata.get("relevance_score", 0.0)) for doc in reranked_docs
                )
            return {"reranked_docs": reranked}
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return state

    def _build_context_node(self, state: ChatState) -> ChatState:
        # Use reranked docs if available, otherwise use retrieved
        reranked = state.get("reranked_docs")
        retrieved = reranked if reranked else state.get("retrieved", [])

        # Label each passage's language when answering in a non-English
        # language, so the model knows which passages it must translate.
        # English context is left exactly as before.
        label_languages = (state.get("answer_language") or languages.DEFAULT.code) != languages.DEFAULT.code

        sections: list[str] = []
        sources: list[SourcePayload] = []

        for i, (doc, score) in enumerate(retrieved, start=1):
            text = (doc.page_content or "").strip()
            if not text:
                continue

            metadata = doc.metadata or {}
            source = str(metadata.get("source") or metadata.get("source_url") or "unknown")
            header = f"[{i}] Source: {source}"
            if label_languages:
                code = str(metadata.get("language") or "")
                lang = languages.LANGUAGES.get(code)
                header += f" (Language: {lang.name if lang else code or 'unknown'})"
            sections.append(f"{header}\n{text}")

            snippet = text.replace("\n", " ")
            if len(snippet) > 240:
                snippet = snippet[:240] + "..."

            sources.append(
                {
                    "source": source,
                    "score": float(score or 0.0),
                    "snippet": snippet,
                }
            )

        return {"context": "\n\n".join(sections), "sources": sources}

    def _answer_node(self, state: ChatState) -> ChatState:
        messages = self._messages(state)
        if not state.get("retrieved"):
            logger.warning("[ANSWER] retrieved is empty — no docs passed retrieval/filtering")
            return {"answer": messages.no_results}

        context = (state.get("context") or "").strip()
        if not context:
            return {"answer": messages.no_text}

        question = state.get("question", "")
        mode = state.get("mode", "normal")
        answer_language = state.get("answer_language") or languages.DEFAULT.code

        if mode == "advanced" and self._ensemble is not None:
            logger.info(f"[ANSWER] using ensemble path for: '{question[:80]}' language={answer_language}")
            try:
                result: EnsembleResult = self._ensemble.run_sync(question, context, answer_language)
                winner = result["winner"]
                answer_text = winner.get("answer") or ""

                if not answer_text:
                    logger.warning("[ENSEMBLE] winner had no answer — falling back to single-shot")
                    return {"answer": self._single_shot_answer(question, context, answer_language)}

                candidates = result.get("candidates", [])
                clusters = result.get("clusters", [])
                total_ms = round(result.get("total_duration_ms", 0), 1)

                for c in candidates:
                    status = "ERROR" if c["error"] else "OK"
                    logger.info(
                        f"[ENSEMBLE]   {c['strategy']:20s} [{status}] "
                        f"{round(c['duration_ms'], 1)}ms"
                    )

                logger.info(
                    f"[ENSEMBLE] winner='{winner.get('strategy')}' "
                    f"votes={winner.get('vote_count', 0)}/{len(candidates)} "
                    f"clusters={[len(c) for c in clusters]} "
                    f"total={total_ms}ms"
                )

                metadata = {
                    "winning_strategy": winner.get("strategy"),
                    "vote_count": winner.get("vote_count", 0),
                    "total_candidates": len(candidates),
                    "cluster_sizes": [len(c) for c in clusters],
                    "total_duration_ms": total_ms,
                    "candidate_strategies": [
                        {
                            "name": c["strategy"],
                            "duration_ms": round(c["duration_ms"], 1),
                            "error": c["error"],
                        }
                        for c in candidates
                    ],
                }

                return {
                    "answer": answer_text,
                    "ensemble_metadata": metadata if self._ensemble_show_metadata else None,
                }

            except Exception as e:
                logger.error(f"[ENSEMBLE] failed — falling back to single-shot answer: {e}")

        logger.info(f"[ANSWER] using single-shot path for: '{question[:80]}' language={answer_language}")
        return {"answer": self._single_shot_answer(question, context, answer_language)}

    @staticmethod
    def _normalize_answer_text(text: str) -> str:
        """Fold model typography that would break the frontend's parsers.

        Safe to apply per-chunk while streaming: every mapping is a single
        codepoint, so no substitution can straddle a chunk boundary.
        """
        return text.translate(_TEXT_NORMALIZATION)

    @staticmethod
    def _answer_inputs(question: str, context: str, answer_language: str) -> dict:
        return {
            "question": question,
            "context": context,
            "language_rules": language_rules(answer_language),
        }

    def _single_shot_answer(self, question: str, context: str, answer_language: str = "en") -> str:
        chain = self._answer_chains.get(answer_language, self._answer_chain)
        response = chain.invoke(self._answer_inputs(question, context, answer_language))
        answer_text = (getattr(response, "content", "") or str(response)).strip()
        return self._normalize_answer_text(answer_text) or "I could not generate a response."

    @staticmethod
    def _require_env(name: str) -> str:
        value = os.getenv(name)
        if not value:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return value

    @staticmethod
    def _read_target_dim() -> int:
        raw = os.getenv("EMBEDDING_DIMENSIONS", "0")
        try:
            return int(raw)
        except ValueError:
            logger.warning(
                f"Invalid EMBEDDING_DIMENSIONS='{raw}'. Using model-native vector size instead."
            )
            return 0

    @staticmethod
    def _read_min_score() -> float:
        raw = os.getenv("RETRIEVAL_MIN_SCORE", "0.5")
        try:
            value = float(raw)
            return max(0.0, min(value, 1.0))
        except ValueError:
            logger.warning(f"Invalid RETRIEVAL_MIN_SCORE='{raw}'. Using 0.5.")
            return 0.5

    @staticmethod
    def _read_threshold(name: str, default: float) -> float:
        raw = os.getenv(name, str(default))
        try:
            value = float(raw)
            return max(0.0, min(value, 1.0))
        except ValueError:
            logger.warning(f"Invalid {name}='{raw}'. Using {default}.")
            return default
