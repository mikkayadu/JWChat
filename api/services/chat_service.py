import os
from typing import TypedDict, List, Tuple, Optional
from enum import Enum

from api.services.ensemble import RAGEnsemble, STRATEGY_MAP, EnsembleResult
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


class SourcePayload(TypedDict):
    source: str
    score: float
    snippet: str


class ChatResult(TypedDict):
    answer: str
    sources: list[SourcePayload]


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
"Return two scores, each a float between 0 and 1:\n"
"- greeting_score: how much the message is purely a greeting or social pleasantry with no "
"information-seeking intent (e.g. 'hi', 'hello', 'thanks', 'how are you', 'good morning'). "
"A message that greets AND asks a real question is NOT a greeting — score it low.\n"
"- out_of_domain_score: how much the message CANNOT be answered from Jehovah's Witness "
"publications and jw.org content. Low = answerable from those sources; high = not.\n\n"

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
"- Tasks unrelated to the knowledge base (translation, coding, summarizing arbitrary text).\n\n"

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

        answer_llm = primary_llm.with_fallbacks(fallback_llms) if fallback_llms else primary_llm
        self._answer_chain = self._answer_prompt | answer_llm

        intent_runnable = primary_llm.with_structured_output(IntentScores)
        if fallback_llms:
            intent_runnable = intent_runnable.with_fallbacks(
                [fb.with_structured_output(IntentScores) for fb in fallback_llms]
            )
        self._intent_chain = self._intent_prompt | intent_runnable

        self._graph = self._build_graph()

        fallback_desc = " → ".join(self._fallback_descs) if self._fallback_descs else "none"
        logger.info(
            f"[CHATSERVICE] ready | llm_provider={self._llm_provider} "
            f"model={self._chat_model or '(provider default)'} "
            f"fallbacks=[{fallback_desc}] "
            f"embeddings_provider={self._embeddings_provider} "
            f"hybrid={self._use_hybrid_search} rerank={self._use_reranking} "
            f"hyde={self._use_hyde} multi_query={self._use_multi_query} "
            f"ensemble={self._use_ensemble}"
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
                template="Please write a detailed passage that could answer the following question:\n{question}"
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

    def _init_fallback_llms(self) -> list[BaseChatModel]:
        """Build an ordered list of fallback chat models, tried in sequence.

        Configured via ``FALLBACKS`` — a comma-separated list of ``provider:model``
        entries, e.g. ``groq:llama-3.1-8b-instant,groq:llama-3.3-70b-versatile``.
        Each entry is split on the FIRST colon only, because model names may
        themselves contain ':' (e.g. ollama ``qwen2.5:1.5b``). A bare ``provider``
        (no colon) uses that provider's default model.

        Falls back to the legacy single ``FALLBACK_LLM_PROVIDER`` /
        ``FALLBACK_CHAT_MODEL`` pair when ``FALLBACKS`` is unset. Entries that fail
        to construct are skipped (logged) so one bad provider can't break startup.
        """
        specs: list[tuple[str, str | None]] = []
        raw = os.getenv("FALLBACKS", "").strip()
        if raw:
            for entry in raw.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                provider, _, model = entry.partition(":")  # split on first ':' only
                specs.append((provider.strip(), model.strip() or None))
        elif os.getenv("FALLBACK_LLM_PROVIDER"):  # legacy single-fallback config
            specs.append((os.getenv("FALLBACK_LLM_PROVIDER"), os.getenv("FALLBACK_CHAT_MODEL")))

        fallbacks: list[BaseChatModel] = []
        for provider, model in specs:
            try:
                fallbacks.append(ModelFactory.create_llm(provider, model=model, temperature=0))
                desc = f"{provider}:{model or '(default)'}"
                self._fallback_descs.append(desc)
                logger.info(f"[CHATSERVICE] fallback registered | {desc}")
            except Exception as e:
                logger.warning(
                    f"Fallback init failed (provider={provider} model={model}): {e} — skipping"
                )
        return fallbacks

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

    def ask(self, question: str, top_k: int = 5, mode: str = "normal") -> ChatResult:
        state = self._graph.invoke({"question": question, "top_k": top_k, "mode": mode})
        result: ChatResult = {
            "answer": state.get("answer", "I could not generate a response."),
            "sources": state.get("sources", []),
        }
        ensemble_metadata = state.get("ensemble_metadata")
        if ensemble_metadata is not None:
            result["ensemble_metadata"] = ensemble_metadata
        return result

    def ask_stream(self, question: str, top_k: int = 5, mode: str = "normal"):
        """Stream a chat answer as transport-agnostic events.

        Yields dicts of the form ``{"event": <name>, "data": <payload>}`` where
        ``event`` is one of:
          * ``sources`` — list[SourcePayload], emitted once before any tokens
          * ``token``   — {"text": str}, one per streamed chunk
          * ``done``    — {}, terminal success marker
          * ``error``   — {"detail": str}, terminal failure marker

        Reuses the existing graph nodes (so retrieval/rerank honour ``mode``), then
        streams the final answer via the fallback-aware answer chain. The ensemble
        path is intentionally not used here — it produces multiple candidates and
        does not stream cleanly; advanced mode degrades to a streamed single shot.
        """
        try:
            state: ChatState = {"question": question, "top_k": top_k, "mode": mode}
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
                yield {"event": "sources", "data": []}
                yield {"event": "token", "data": {"text": canned["answer"]}}
                yield {"event": "done", "data": {}}
                return

            # In-domain: mirror the graph edges (retrieve → rerank → build_context).
            state.update(self._retrieve_node(state))
            state.update(self._rerank_node(state))
            state.update(self._build_context_node(state))

            if not state.get("retrieved"):
                logger.warning("[STREAM] retrieved is empty — no docs passed retrieval/filtering")
                yield {"event": "sources", "data": []}
                yield {"event": "token", "data": {"text": "I could not find relevant content in the indexed books yet."}}
                yield {"event": "done", "data": {}}
                return

            context = (state.get("context") or "").strip()
            if not context:
                yield {"event": "sources", "data": []}
                yield {"event": "token", "data": {"text": "I found matches, but they did not contain readable text content."}}
                yield {"event": "done", "data": {}}
                return

            yield {"event": "sources", "data": state.get("sources", [])}

            streamed_any = False
            for chunk in self._answer_chain.stream({"question": question, "context": context}):
                text = getattr(chunk, "content", "") or ""
                if text:
                    streamed_any = True
                    yield {"event": "token", "data": {"text": text}}

            if not streamed_any:
                yield {"event": "token", "data": {"text": "I could not generate a response."}}

            yield {"event": "done", "data": {}}

        except Exception as e:
            logger.error(f"[STREAM] failed: {e}")
            yield {
                "event": "error",
                "data": {
                    "detail": (
                        "The AI model is currently unavailable, so I can't process your "
                        "question right now. Please try again in a moment."
                    )
                },
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
        if not question:
            return {"intent": "greeting", "greeting_score": 1.0, "out_of_domain_score": 0.0}

        try:
            scores = self._intent_chain.invoke({"question": question})
            greeting_score = float(scores.greeting_score)
            out_of_domain_score = float(scores.out_of_domain_score)
            logger.info(
                f"[INTENT] q='{question[:60]}' greeting={greeting_score:.2f} "
                f"out_of_domain={out_of_domain_score:.2f} "
                f"(thresholds greeting>={self._greeting_threshold} ood>={self._out_of_domain_threshold})"
            )
        except Exception as e:
            # The classifier LLM (and its fallback) failed — this is a service
            # outage, NOT an off-topic question. Route to a clear "unavailable"
            # message instead of the misleading out-of-domain reply.
            logger.warning(f"Intent classification failed (model/service unavailable): {e}")
            return {"intent": "error", "greeting_score": 0.0, "out_of_domain_score": 0.0}

        if greeting_score >= self._greeting_threshold:
            return {
                "intent": "greeting",
                "greeting_score": greeting_score,
                "out_of_domain_score": out_of_domain_score,
            }

        if out_of_domain_score >= self._out_of_domain_threshold:
            return {
                "intent": "out_of_domain",
                "greeting_score": greeting_score,
                "out_of_domain_score": out_of_domain_score,
            }

        return {
            "intent": "in_domain",
            "greeting_score": greeting_score,
            "out_of_domain_score": out_of_domain_score,
        }

    @staticmethod
    def _route_intent(state: ChatState) -> str:
        return str(state.get("intent") or "unknown")

    @staticmethod
    def _greeting_response_node(_: ChatState) -> ChatState:
        return {
            "answer": (
                "Hello. I can help with questions based on indexed Jehovah's Witnesses content. "
                "Ask me a question about JW teachings, publications, or material from jw.org."
            ),
            "sources": [],
        }

    @staticmethod
    def _out_of_domain_response_node(_: ChatState) -> ChatState:
        return {
            "answer": (
                "I can only answer from indexed Jehovah's Witnesses sources. "
                "Please ask a JW-related question so I can cite the relevant material."
            ),
            "sources": [],
        }

    @staticmethod
    def _service_unavailable_node(_: ChatState) -> ChatState:
        return {
            "answer": (
                "The AI model is currently unavailable, so I can't process your question right "
                "now. Please try again in a moment."
            ),
            "sources": [],
        }

    def _retrieve_node(self, state: ChatState) -> ChatState:
        question = (state.get("question") or "").strip()
        if not question:
            return {"retrieved": []}

        mode = state.get("mode", "normal")
        top_k = int(state.get("top_k", 5))
        top_k = max(1, min(top_k, 20))
        use_reranking = mode in ("thinking", "advanced") and self._reranker is not None
        use_multi_query = mode in ("thinking", "advanced") and self._multi_query_chain is not None
        fetch_k = self._retrieval_k if use_reranking else top_k * 2

        logger.info(f"[RETRIEVE] question='{question}' mode={mode} fetch_k={fetch_k} reranking={use_reranking} multi_query={use_multi_query}")

        queries = [question]
        if use_multi_query:
            try:
                queries = self._expand_queries(question)
                logger.info(f"[MULTI-QUERY] {len(queries)} queries total:")
                for i, q in enumerate(queries):
                    logger.info(f"  [{i}] {q}")
            except Exception as e:
                logger.warning(f"Multi-query expansion failed, using original query: {e}")

        # Step 2: HyDE — rewrite the primary query as a hypothetical answer passage
        hyde_doc = None
        if self._hyde_chain and self._use_hyde:
            try:
                hyde_doc = self._hyde_chain.invoke({"question": question})["text"]
                queries[0] = hyde_doc
                logger.info(f"[HYDE] hypothetical passage (first 120 chars): {hyde_doc[:120]}")
            except Exception as e:
                logger.warning(f"HyDE failed, using original query: {e}")

        # Step 3: Fetch for every query variant; deduplicate by content fingerprint
        seen: dict[str, tuple[Document, float]] = {}
        for query in queries:
            hits = self._fetch_from_pinecone(query, fetch_k)
            logger.info(f"[PINECONE] query='{query[:60]}...' → {len(hits)} hits passed filter")
            for doc, score in hits:
                key = doc.page_content[:120]
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

        # Step 4: BM25 hybrid re-scoring — combine dense scores with keyword scores
        if retrieved:
            retrieved = self._apply_bm25_fusion(question, retrieved)
            logger.info(f"[BM25] top-3 fused scores: {[round(s,3) for _,s in retrieved[:5]]}")

        return {
            "retrieved": retrieved,
            "hyde_doc": hyde_doc,
            "query_variants": queries[1:],
        }

    def _fetch_from_pinecone(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Embed query and fetch top-k matches from Pinecone above the min score threshold."""
        query_vector = self._embeddings.embed_query(query)
        result = self._index.query(vector=query_vector, top_k=k, include_metadata=True)
        matches = result.get("matches", [])
        min_score = self._read_min_score()

        all_scores = [float(m.get("score") or 0.0) for m in matches]
        logger.info(f"[PINECONE RAW] {len(matches)} matches, min_score_filter={min_score}, scores={[round(s,3) for s in all_scores[:10]]}")

        retrieved: list[tuple[Document, float]] = []
        for match in matches:
            metadata = match.get("metadata") or {}
            score = float(match.get("score") or 0.0)
            if score < min_score:
                continue
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
            tokenized_corpus = [doc.page_content.lower().split() for doc in docs]
            bm25 = BM25Okapi(tokenized_corpus)
            raw_scores = bm25.get_scores(question.lower().split())

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

    def _rerank_node(self, state: ChatState) -> ChatState:
        mode = state.get("mode", "normal")
        if mode == "normal" or not self._reranker:
            return state

        retrieved = state.get("retrieved", [])
        if not retrieved:
            return state

        try:
            docs = [doc for doc, score in retrieved]
            query = state.get("question", "")

            # PineconeRerank expects documents and query
            reranked_docs = self._reranker.compress_documents(docs, query)

            # Convert back to retrieved format with new scores
            reranked = [(doc, doc.metadata.get("relevance_score", 0.0)) for doc in reranked_docs]
            return {"reranked_docs": reranked}
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return state

    def _build_context_node(self, state: ChatState) -> ChatState:
        # Use reranked docs if available, otherwise use retrieved
        reranked = state.get("reranked_docs")
        retrieved = reranked if reranked else state.get("retrieved", [])

        sections: list[str] = []
        sources: list[SourcePayload] = []

        for i, (doc, score) in enumerate(retrieved, start=1):
            text = (doc.page_content or "").strip()
            if not text:
                continue

            metadata = doc.metadata or {}
            source = str(metadata.get("source") or metadata.get("source_url") or "unknown")
            sections.append(f"[{i}] Source: {source}\n{text}")

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
        if not state.get("retrieved"):
            logger.warning("[ANSWER] retrieved is empty — no docs passed retrieval/filtering")
            return {"answer": "I could not find relevant content in the indexed books yet."}

        context = (state.get("context") or "").strip()
        if not context:
            return {"answer": "I found matches, but they did not contain readable text content."}

        question = state.get("question", "")
        mode = state.get("mode", "normal")

        if mode == "advanced" and self._ensemble is not None:
            logger.info(f"[ANSWER] using ensemble path for: '{question[:80]}'")
            try:
                result: EnsembleResult = self._ensemble.run_sync(question, context)
                winner = result["winner"]
                answer_text = winner.get("answer") or ""

                if not answer_text:
                    logger.warning("[ENSEMBLE] winner had no answer — falling back to single-shot")
                    return {"answer": self._single_shot_answer(question, context)}

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

        logger.info(f"[ANSWER] using single-shot path for: '{question[:80]}'")
        return {"answer": self._single_shot_answer(question, context)}

    def _single_shot_answer(self, question: str, context: str) -> str:
        response = self._answer_chain.invoke({"question": question, "context": context})
        answer_text = (getattr(response, "content", "") or str(response)).strip()
        return answer_text or "I could not generate a response."

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
