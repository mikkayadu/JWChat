import json
from pathlib import Path
from typing import Iterator

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import languages
from api.services.chat_service import ChatService
from api.services.scripture_service import (
    ScriptureNotFound,
    get_verses,
)

load_dotenv()

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="normal", pattern="^(normal|thinking|advanced)$")
    # "auto" detects the language from the question; a code forces it. Left
    # unset by frontends that predate languages, which then get exactly the
    # old response stream (no `meta` event).
    language: str | None = None

    @field_validator("language")
    @classmethod
    def _check_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().lower()
        allowed = ["auto"] + [lang.code for lang in languages.enabled()]
        if value not in allowed:
            raise ValueError(f"language must be one of {allowed}")
        return value


class SourceItem(BaseModel):
    source: str
    score: float
    snippet: str


class EnsembleCandidateInfo(BaseModel):
    name: str
    duration_ms: float
    error: bool


class EnsembleMetadata(BaseModel):
    winning_strategy: str | None
    vote_count: int
    total_candidates: int
    cluster_sizes: list[int]
    total_duration_ms: float
    candidate_strategies: list[EnsembleCandidateInfo]


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    ensemble_metadata: EnsembleMetadata | None = None
    answer_language: str = "en"
    used_fallback: bool = False


app = FastAPI(title="Book Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_chat_service: ChatService | None = None

def _get_chat_service() -> ChatService:
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service


@app.get("/")
def home() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.api_route("/health", methods=["GET", "HEAD"])
def health() -> dict[str, str]:
    return {"status": "ok"}


class VerseItem(BaseModel):
    verse: int
    text: str


class ScriptureResponse(BaseModel):
    reference: str
    url: str
    verses: list[VerseItem]


class LanguageItem(BaseModel):
    code: str
    native_name: str
    book_names: list[str]


@app.get("/api/languages", response_model=list[LanguageItem])
def list_languages() -> list[LanguageItem]:
    """Enabled answer languages, with each one's 66 Bible book names (index =
    book number - 1) so the frontend can link scripture references without
    keeping its own book tables."""
    return [
        LanguageItem(code=lang.code, native_name=lang.native_name, book_names=list(lang.book_names))
        for lang in languages.enabled()
    ]


@app.get("/api/scripture", response_model=ScriptureResponse)
def scripture(
    chapter: int,
    book: int,
    verse: int,
    end_verse: int | None = None,
    lang: str = "en",
    slug: str | None = None,
) -> ScriptureResponse:
    """Fetch verse text from jw.org for inline display.

    Params come pre-parsed from the frontend: `book` (1-66 book number),
    `chapter`, `verse`, optional `end_verse`, and `lang` (an enabled language
    code, default en). `slug` is accepted for older frontends but unused:
    jw.org's finder resolves the book number in every language.
    """
    if not (1 <= book <= 66) or chapter < 1 or verse < 1:
        raise HTTPException(status_code=400, detail="Invalid scripture coordinates")
    lang = lang.strip().lower()
    if lang not in {enabled.code for enabled in languages.enabled()}:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {lang}")

    try:
        url, verses = get_verses(lang, book, chapter, verse, end_verse)
    except ScriptureNotFound as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Could not reach jw.org: {e}") from e

    last = end_verse if end_verse and end_verse > verse else None
    reference = f"{chapter}:{verse}" + (f"-{last}" if last else "")
    return ScriptureResponse(reference=reference, url=url, verses=[VerseItem(**v) for v in verses])


_og_image_cache: dict[str, str | None] = {}

@app.get("/api/og-image")
def og_image(url: str) -> dict[str, str | None]:
    if url in _og_image_cache:
        return {"image": _og_image_cache[url]}

    image_url: str | None = None
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=8, headers={
            "User-Agent": "Mozilla/5.0 (compatible; JWChat/1.0)",
        })
        resp.raise_for_status()
        html = resp.text

        import re
        match = re.search(r'<meta\s+property=["\']og:image["\']\s+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
        if not match:
            match = re.search(r'<meta\s+content=["\']([^"\']+)["\']\s+property=["\']og:image["\']', html, re.IGNORECASE)
        if match:
            image_url = match.group(1)
    except Exception:
        pass

    _og_image_cache[url] = image_url
    return {"image": image_url}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        result = _get_chat_service().ask(req.question, req.top_k, req.mode, req.language or "auto")
        ensemble_meta = result.get("ensemble_metadata")
        return ChatResponse(
            answer=result["answer"],
            sources=[SourceItem(**source) for source in result["sources"]],
            ensemble_metadata=EnsembleMetadata(**ensemble_meta) if ensemble_meta else None,
            answer_language=result.get("answer_language", "en"),
            used_fallback=result.get("used_fallback", False),
        )

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat request failed: {e}") from e


def _sse(event: str, data: object) -> str:
    """Format a single Server-Sent Events frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Stream the answer token-by-token over Server-Sent Events.

    Emits a `sources` event, then a series of `token` events, then `done`
    (or `error`). When the request includes `language`, a `meta` event with
    `answer_language` and `used_fallback` comes first. Generation runs on the
    primary LLM and transparently falls back to the configured backup on any
    failure.
    """

    def event_stream() -> Iterator[str]:
        try:
            events = _get_chat_service().ask_stream(
                req.question,
                req.top_k,
                req.mode,
                language=req.language or "auto",
                emit_meta=req.language is not None,
            )
            for event in events:
                yield _sse(event["event"], event["data"])
        except Exception as e:  # safety net — ask_stream already emits `error`
            yield _sse("error", {"detail": f"Chat request failed: {e}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering so tokens flush live
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
