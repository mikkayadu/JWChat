import os
from typing import TypedDict, List, Tuple, Optional
from enum import Enum

from api.services.ensemble import RAGEnsemble, STRATEGY_MAP, EnsembleResult
from ingestion.utils.embedder import normalize_vector_dimensions
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
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
    intent: str
    greeting_score: float
    out_of_domain_score: float
    retrieved: list[tuple[Document, float]]
    context: str
    sources: list[SourcePayload]
    answer: str
    # Advanced retrieval fields
    reranked_docs: list[tuple[Document, float]]  # After reranking
    query_variants: list[str]  # For multi-query expansion
    hyde_doc: str  # Hypothetical document for HyDE
    ensemble_metadata: dict | None  # Ensemble result metadata


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


class DimensionAwareGeminiEmbeddings(Embeddings):
    """Wrap Gemini embeddings so vectors always match Pinecone index dimension."""

    def __init__(self, model: str, google_api_key: str, target_dim: int) -> None:
        self._target_dim = target_dim
        self._base = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=google_api_key,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._base.embed_documents(texts)
        if self._target_dim <= 0:
            return vectors
        return [normalize_vector_dimensions(v, self._target_dim) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._base.embed_query(text)
        if self._target_dim <= 0:
            return vector
        return normalize_vector_dimensions(vector, self._target_dim)


class ChatService:
    def __init__(self) -> None:
        self._gemini_key = self._require_env("GEMINI_API_KEY")
        self._pinecone_key = self._require_env("PINECONE_API_KEY")
        self._index_name = os.getenv("PINECONE_INDEX", "website-rag")
        self._embedding_model = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
        self._chat_model = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
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

        self._embeddings = DimensionAwareGeminiEmbeddings(
            model=self._embedding_model,
            google_api_key=self._gemini_key,
            target_dim=self._target_dim,
        )
        self._pinecone_client = Pinecone(api_key=self._pinecone_key)
        self._index = self._pinecone_client.Index(self._index_name)

        # Initialize advanced retrieval components
        self._reranker = self._init_reranker() if self._use_reranking else None
        self._hyde_chain = self._init_hyde() if self._use_hyde else None
        self._multi_query_chain = self._init_multi_query() if self._use_multi_query else None

        self._answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a knowledgeable assistant specializing in Jehovah's Witness teachings, "
                    "beliefs, and publications — drawing exclusively from their official books, magazines, "
                    "and website content (e.g., jw.org).\n\n"

                    "STRICT RULES:\n"
                    "1. Answer ONLY from the provided context. Do not use outside knowledge or inference.\n"
                    "2. If the answer is not found in the context, respond exactly: "
                    "'I could not find information about this in the provided sources.'\n"
                    "3. Cite every factual claim with inline source references like [1], [2], etc.\n"
                    "4. Explain and restate the source content clearly in your own words — "
                    "but never introduce claims, conclusions, or implications not present in the source.\n\n"

                    "RESPONSE FORMAT:\n"
                    "- Be concise and direct.\n"
                    "- Place citations immediately after the sentence they support.\n"
                    "- If multiple sources support a point, list all relevant citations: [1][3].\n"
                    "- End with a 'Sources:' section listing each cited reference by number."
                    
                    
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
                    "You are an intent classifier for a Jehovah's Witness domain assistant. "
                    "Return scores between 0 and 1. "
                    "greeting_score: how much the message is primarily greeting/social. "
                    "out_of_domain_score: how much the message is not answerable from Jehovah's Witness publications and jw.org content.",
                ),
                (
                    "human",
                    "Question:\n{question}",
                ),
            ]
        )

        llm = ChatGoogleGenerativeAI(
            model=self._chat_model,
            google_api_key=self._gemini_key,
            temperature=0,
        )
        self._ensemble = self._init_ensemble(llm) if self._use_ensemble else None
        self._answer_chain = self._answer_prompt | llm
        self._intent_chain = self._intent_prompt | llm.with_structured_output(IntentScores)
        self._graph = self._build_graph()

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
            llm = ChatGoogleGenerativeAI(
                model=self._chat_model,
                google_api_key=self._gemini_key,
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
            return ChatGoogleGenerativeAI(
                model=self._chat_model,
                google_api_key=self._gemini_key,
                temperature=0.3,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize multi-query LLM: {e}")
            return None

    def _init_ensemble(self, llm: ChatGoogleGenerativeAI) -> RAGEnsemble | None:
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
                f"disabling ensemble"
            )
            self._use_ensemble = False
            return None

        logger.info(f"[ENSEMBLE] initialized with {len(strategies)} strategies: {list(strategies.keys())}")
        return RAGEnsemble(
            strategies=strategies,
            embeddings=self._embeddings,
            similarity_threshold=self._ensemble_similarity_threshold,
        )

    def ask(self, question: str, top_k: int = 5) -> ChatResult:
        state = self._graph.invoke({"question": question, "top_k": top_k})
        result: ChatResult = {
            "answer": state.get("answer", "I could not generate a response."),
            "sources": state.get("sources", []),
        }
        ensemble_metadata = state.get("ensemble_metadata")
        if ensemble_metadata is not None:
            result["ensemble_metadata"] = ensemble_metadata
        return result

    def _build_graph(self):
        graph = StateGraph(ChatState)
        graph.add_node("classify_intent", self._classify_intent_node)
        graph.add_node("greeting_response", self._greeting_response_node)
        graph.add_node("out_of_domain_response", self._out_of_domain_response_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("build_context", self._build_context_node)
        graph.add_node("answer", self._answer_node)

        # Add reranking node if enabled
        if self._reranker and self._use_reranking:
            graph.add_node("rerank", self._rerank_node)
            logger.info("Reranking node added to graph")

        graph.set_entry_point("classify_intent")
        graph.add_conditional_edges(
            "classify_intent",
            self._route_intent,
            {
                "greeting": "greeting_response",
                "out_of_domain": "out_of_domain_response",
                "in_domain": "retrieve",
                "unknown": "out_of_domain_response",
            },
        )

        graph.add_edge("greeting_response", END)
        graph.add_edge("out_of_domain_response", END)

        # Route through reranking if enabled
        if self._reranker and self._use_reranking:
            graph.add_edge("retrieve", "rerank")
            graph.add_edge("rerank", "build_context")
        else:
            graph.add_edge("retrieve", "build_context")

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
        except Exception as e:
            logger.warning(f"Intent classification failed, defaulting to safe fallback response: {e}")
            return {"intent": "unknown", "greeting_score": 0.0, "out_of_domain_score": 0.0}

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

    def _retrieve_node(self, state: ChatState) -> ChatState:
        question = (state.get("question") or "").strip()
        if not question:
            return {"retrieved": []}

        top_k = int(state.get("top_k", 5))
        top_k = max(1, min(top_k, 20))
        fetch_k = self._retrieval_k if self._use_reranking else top_k * 2

        logger.info(f"[RETRIEVE] question='{question}' fetch_k={fetch_k} reranking={self._use_reranking} hybrid={self._use_hybrid_search}")

        # Step 1: Build query list — start with original, expand via multi-query
        queries = [question]
        if self._multi_query_chain and self._use_multi_query:
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
        logger.info(f"[RETRIEVE] {len(retrieved)} unique docs after dedup")

        # Step 4: BM25 hybrid re-scoring — combine dense scores with keyword scores
        if self._use_hybrid_search and retrieved:
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
        """Rerank retrieved documents for better precision."""
        if not self._reranker or not self._use_reranking:
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

        if self._use_ensemble and self._ensemble is not None:
            try:
                result: EnsembleResult = self._ensemble.run_sync(question, context)
                winner = result["winner"]
                answer_text = winner.get("answer") or ""

                if not answer_text:
                    logger.warning("[ENSEMBLE] winner had no answer — falling back to single-shot")
                    return {"answer": self._single_shot_answer(question, context)}

                metadata = {
                    "winning_strategy": winner.get("strategy"),
                    "vote_count": winner.get("vote_count", 0),
                    "total_candidates": len(result.get("candidates", [])),
                    "cluster_sizes": [len(c) for c in result.get("clusters", [])],
                    "total_duration_ms": round(result.get("total_duration_ms", 0), 1),
                    "candidate_strategies": [
                        {
                            "name": c["strategy"],
                            "duration_ms": round(c["duration_ms"], 1),
                            "error": c["error"],
                        }
                        for c in result.get("candidates", [])
                    ],
                }

                return {
                    "answer": answer_text,
                    "ensemble_metadata": metadata if self._ensemble_show_metadata else None,
                }

            except Exception as e:
                logger.error(f"[ENSEMBLE] failed — falling back to single-shot answer: {e}")

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
