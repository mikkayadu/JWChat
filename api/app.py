import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)


class SourceItem(BaseModel):
    source: str
    score: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


app = FastAPI(title="Book Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {var_name}")
    return value


def _embed_query(query: str) -> list[float]:
    import google.generativeai as genai

    gemini_key = _require_env("GEMINI_API_KEY")
    model_name = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

    genai.configure(api_key=gemini_key)
    emb = genai.embed_content(
        model=model_name,
        content=query,
        task_type="retrieval_query",
    )
    return emb["embedding"]


def _query_pinecone(query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
    from pinecone import Pinecone

    pinecone_key = _require_env("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "website-rag")

    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(index_name)

    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    return result.get("matches", [])


def _build_context(matches: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    for i, match in enumerate(matches, start=1):
        metadata = match.get("metadata") or {}
        text = (metadata.get("text") or "").strip()
        source = metadata.get("source") or metadata.get("source_url") or "unknown"
        if not text:
            continue
        sections.append(f"[{i}] Source: {source}\n{text}")

    return "\n\n".join(sections)


def _answer_with_gemini(question: str, context: str) -> str:
    import google.generativeai as genai

    gemini_key = _require_env("GEMINI_API_KEY")
    genai.configure(api_key=gemini_key)

    model_name = os.getenv("CHAT_MODEL", "gemini-1.5-flash")
    model = genai.GenerativeModel(model_name)

    prompt = (
        "You are a helpful book assistant. Answer only from the provided context. "
        "If the context does not contain the answer, say so clearly. "
        "Always include short source citations like [1], [2] tied to the context sections.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Return a concise answer with citations."
    )

    response = model.generate_content(prompt)
    return (response.text or "I could not generate a response.").strip()


@app.get("/")
def home() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        query_vector = _embed_query(req.question)
        matches = _query_pinecone(query_vector, req.top_k)

        if not matches:
            return ChatResponse(
                answer="I could not find relevant content in the indexed books yet.",
                sources=[],
            )

        context = _build_context(matches)
        if not context:
            return ChatResponse(
                answer="I found matches, but they did not contain readable text content.",
                sources=[],
            )

        answer = _answer_with_gemini(req.question, context)

        sources: list[SourceItem] = []
        for match in matches:
            metadata = match.get("metadata") or {}
            snippet = (metadata.get("text") or "").strip().replace("\n", " ")
            snippet = snippet[:240] + ("..." if len(snippet) > 240 else "")
            source = metadata.get("source") or metadata.get("source_url") or "unknown"
            score = float(match.get("score") or 0.0)
            sources.append(SourceItem(source=source, score=score, snippet=snippet))

        return ChatResponse(answer=answer, sources=sources)

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat request failed: {e}") from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("chatbot_api:app", host="0.0.0.0", port=8000, reload=True)
