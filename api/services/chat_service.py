import os
from typing import TypedDict, List, Tuple, Optional
from enum import Enum

from ingestion.utils.embedder import normalize_vector_dimensions
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import END, StateGraph
from loguru import logger
from pinecone import Pinecone
from pydantic import BaseModel, Field

# Optional imports for advanced features
try:
    from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
    from langchain_community.retrievers import BM25Retriever
    from langchain_pinecone import PineconeRerank
    ADVANCED_RETRIEVAL_AVAILABLE = True
except ImportError:
    ADVANCED_RETRIEVAL_AVAILABLE = False
    logger.warning("Advanced retrieval features not available. Install: langchain-pinecone, langchain-community")

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
    reranked_docs: list[Document]  # After reranking
    query_variants: list[str]  # For multi-query expansion
    hyde_doc: str  # Hypothetical document for HyDE


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
        self._score_threshold = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.75"))
        self._rerank_top_n = int(os.getenv("RERANK_TOP_N", "5"))
        self._retrieval_k = int(os.getenv("RETRIEVAL_K", "20"))  # Initial retrieval count

        self._embeddings = DimensionAwareGeminiEmbeddings(
            model=self._embedding_model,
            google_api_key=self._gemini_key,
            target_dim=self._target_dim,
        )
        self._index = Pinecone(api_key=self._pinecone_key).Index(self._index_name)

        # Initialize advanced retrieval components
        self._retriever = self._init_retriever()
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
                    "4. Never speculate, paraphrase beyond what the source supports, or fill gaps.\n\n"

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
        self._answer_chain = self._answer_prompt | llm
        self._intent_chain = self._intent_prompt | llm.with_structured_output(IntentScores)
        self._graph = self._build_graph()

    def _init_retriever(self):
        """Initialize the retriever with optional hybrid search."""
        if not ADVANCED_RETRIEVAL_AVAILABLE or not self._use_hybrid_search:
            logger.info("Using basic vector retriever")
            return None  # Will use direct Pinecone query

        try:
            from langchain_pinecone import PineconeVectorStore
            
            vectorstore = PineconeVectorStore(
                index=self._index,
                embedding=self._embeddings,
                text_key="text"
            )
            
            # Dense retriever
            dense_retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self._retrieval_k,
                    "score_threshold": self._score_threshold
                }
            )
            
            # TODO: Add BM25 sparse retriever for full hybrid search
            # This requires document store setup
            
            return dense_retriever
        except Exception as e:
            logger.warning(f"Failed to initialize advanced retriever: {e}")
            return None

    def _init_reranker(self):
        """Initialize the reranker for improved precision."""
        if not ADVANCED_RETRIEVAL_AVAILABLE:
            return None
        try:
            return PineconeRerank(
                model="bge-reranker-v2-m3",
                top_n=self._rerank_top_n
            )
        except Exception as e:
            logger.warning(f"Failed to initialize reranker: {e}")
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
        """Initialize multi-query expansion chain."""
        if not ADVANCED_RETRIEVAL_AVAILABLE or not HYDE_AVAILABLE:
            return None
        try:
            llm = ChatGoogleGenerativeAI(
                model=self._chat_model,
                google_api_key=self._gemini_key,
                temperature=0.3,
            )
            # Will implement manual multi-query in retrieve node
            return llm
        except Exception as e:
            logger.warning(f"Failed to initialize multi-query: {e}")
            return None

    def ask(self, question: str, top_k: int = 5) -> ChatResult:
        state = self._graph.invoke({"question": question, "top_k": top_k})
        return {
            "answer": state.get("answer", "I could not generate a response."),
            "sources": state.get("sources", []),
        }

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

        # Apply query transformation if enabled
        query_for_retrieval = question
        hyde_doc = None

        if self._hyde_chain and self._use_hyde:
            try:
                hyde_doc = self._hyde_chain.invoke({"question": question})["text"]
                query_for_retrieval = hyde_doc
                logger.debug(f"HyDE generated hypothetical document")
            except Exception as e:
                logger.warning(f"HyDE failed, using original query: {e}")

        # Use advanced retriever if available
        if self._retriever and self._use_hybrid_search:
            try:
                docs = self._retriever.invoke(query_for_retrieval)
                retrieved = [(doc, 1.0) for doc in docs]  # Score not available from retriever
                return {
                    "retrieved": retrieved,
                    "hyde_doc": hyde_doc,
                }
            except Exception as e:
                logger.warning(f"Advanced retriever failed, falling back to direct query: {e}")

        # Fallback: direct Pinecone query
        query_vector = self._embeddings.embed_query(query_for_retrieval)
        result = self._index.query(
            vector=query_vector,
            top_k=self._retrieval_k if self._use_reranking else top_k,
            include_metadata=True,
        )

        min_score = self._read_min_score()

        retrieved: list[tuple[Document, float]] = []
        for match in result.get("matches", []):
            metadata = match.get("metadata") or {}
            page_content = str(metadata.get("text") or "")
            score = float(match.get("score") or 0.0)

            if score < min_score:
                continue

            retrieved.append(
                (
                    Document(page_content=page_content, metadata=metadata),
                    score,
                )
            )

        return {"retrieved": retrieved, "hyde_doc": hyde_doc}

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
            return {"answer": "I could not find relevant content in the indexed books yet."}

        context = (state.get("context") or "").strip()
        if not context:
            return {"answer": "I found matches, but they did not contain readable text content."}

        question = state.get("question", "")
        response = self._answer_chain.invoke({"question": question, "context": context})
        answer_text = (getattr(response, "content", "") or str(response)).strip()

        if not answer_text:
            answer_text = "I could not generate a response."

        return {"answer": answer_text}

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
        raw = os.getenv("RETRIEVAL_MIN_SCORE", "0.0")
        try:
            value = float(raw)
            return max(0.0, min(value, 1.0))
        except ValueError:
            logger.warning(f"Invalid RETRIEVAL_MIN_SCORE='{raw}'. Using 0.0.")
            return 0.0

    @staticmethod
    def _read_threshold(name: str, default: float) -> float:
        raw = os.getenv(name, str(default))
        try:
            value = float(raw)
            return max(0.0, min(value, 1.0))
        except ValueError:
            logger.warning(f"Invalid {name}='{raw}'. Using {default}.")
            return default
