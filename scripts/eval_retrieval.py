"""
Retrieval eval gate for a new language
──────────────────────────────────────
Measures whether the embedding model retrieves the right pages for questions
asked in a language, before that language is switched on for users.

For every line of the questions file it embeds the question, queries that
language's Pinecone namespace, and checks whether any expected URL comes back
in the top k. It reports hit@k, MRR and score percentiles; the percentiles are
what the language's ``min_score`` in languages.py should be set from.

With ``--check-classifier`` it also sends each question to the intent
classifier's primary model and to every INTENT_FALLBACKS model with the
detect-and-translate instructions the chat API will use, and prints the
results for a Twi speaker to spot-check. This uses a standalone prompt: the
classifier itself only learns these fields in the query-time PR.

Questions file (JSONL, one object per line):
    {"question": "...", "expected_urls": ["https://www.jw.org/tw/..."]}
Write questions from pages that are actually in the ingested sample, otherwise
the gate measures crawl coverage rather than embedding quality.

Usage (point PINECONE_INDEX at the dev index):
    PINECONE_INDEX=jwchat-dev python -m scripts.eval_retrieval \\
        --questions eval/tw_questions.jsonl --language tw --check-classifier
"""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import languages  # noqa: E402

load_dotenv()

DETECT_TRANSLATE_INSTRUCTIONS = (
    "Identify the language of the user's question and give its ISO 639-1 code "
    "(use 'tw' for Twi/Akan, including Twi typed without the letters ɛ and ɔ). "
    "Then translate the question into natural English for searching English "
    "documents; if it is already English, return it unchanged. Keep names of "
    "people, places and Bible books recognisable."
)


def normalise_url(url: str) -> str:
    """Compare URLs by host and path only: ignore scheme, fragment, query and
    a trailing slash, so ingested and hand-written URLs match."""
    parsed = urlparse((url or "").strip())
    host = (parsed.hostname or "").lower()
    path = parsed.path.rstrip("/") or "/"
    return f"{host}{path}"


def first_hit_rank(retrieved_urls: list[str], expected_urls: list[str]) -> int | None:
    """1-based rank of the first retrieved URL that is expected, or None."""
    expected = {normalise_url(u) for u in expected_urls}
    for rank, url in enumerate(retrieved_urls, start=1):
        if normalise_url(url) in expected:
            return rank
    return None


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile; ``pct`` in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def summarise(ranks: list[int | None], k: int) -> dict[str, float]:
    n = len(ranks) or 1
    hits = sum(1 for r in ranks if r is not None and r <= k)
    mrr = sum(1 / r for r in ranks if r is not None and r <= k) / n
    return {f"hit@{k}": hits / n, "mrr": mrr}


def load_questions(path: Path) -> list[dict]:
    items = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not item.get("question") or not item.get("expected_urls"):
            raise ValueError(f"{path}:{line_no}: needs 'question' and 'expected_urls'")
        items.append(item)
    return items


def _source_url(metadata: dict) -> str:
    return str(metadata.get("source") or metadata.get("source_url") or "")


def run_retrieval(items: list[dict], lang: languages.Language, k: int, fetch_k: int) -> dict:
    from pinecone import Pinecone

    from api.services.providers import ModelFactory

    embeddings = ModelFactory.create_embeddings(
        os.getenv("EMBEDDINGS_PROVIDER", "google"),
        model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
        target_dim=int(os.getenv("EMBEDDING_DIMENSIONS", "0") or 0),
    )
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(
        os.getenv("PINECONE_INDEX", "website-rag")
    )

    ranks: list[int | None] = []
    top1_scores: list[float] = []
    all_scores: list[float] = []
    hit_scores: list[float] = []

    for item in items:
        vector = embeddings.embed_query(item["question"])
        result = index.query(vector=vector, top_k=fetch_k, include_metadata=True, **lang.namespace_kwargs)
        matches = result.get("matches", [])
        urls = [_source_url(m.get("metadata") or {}) for m in matches]
        scores = [float(m.get("score") or 0.0) for m in matches]

        rank = first_hit_rank(urls, item["expected_urls"])
        ranks.append(rank)
        all_scores.extend(scores)
        if scores:
            top1_scores.append(scores[0])
        if rank is not None:
            hit_scores.append(scores[rank - 1])

        status = f"hit@{rank}" if rank is not None and rank <= k else "MISS"
        top = f"{scores[0]:.3f}" if scores else "-"
        print(f"[{status:>6}] top1={top} {item['question'][:70]}")

    return {
        "ranks": ranks,
        "metrics": summarise(ranks, k),
        "top1_scores": top1_scores,
        "all_scores": all_scores,
        "hit_scores": hit_scores,
    }


def run_classifier_check(items: list[dict]) -> None:
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field

    from api.services.providers import ModelFactory

    class DetectTranslate(BaseModel):
        detected_language: str = Field(description="ISO 639-1 code of the question's language")
        english_query: str = Field(description="The question translated into English")

    prompt = ChatPromptTemplate.from_messages(
        [("system", DETECT_TRANSLATE_INSTRUCTIONS), ("human", "Question:\n{question}")]
    )

    specs = [(os.getenv("INTENT_LLM_PROVIDER", "google"), os.getenv("INTENT_CHAT_MODEL"))]
    for entry in os.getenv("INTENT_FALLBACKS", "").split(","):
        if entry.strip():
            provider, _, model = entry.strip().partition(":")
            specs.append((provider, model or None))

    for provider, model in specs:
        label = f"{provider}:{model or '(default)'}"
        print(f"\n=== classifier check: {label}")
        try:
            chain = prompt | ModelFactory.create_llm(provider, model=model, temperature=0).with_structured_output(
                DetectTranslate
            )
        except Exception as e:
            print(f"  could not build model: {e}")
            continue
        for item in items:
            try:
                out = chain.invoke({"question": item["question"]})
                print(f"  [{out.detected_language}] {item['question'][:50]}\n      -> {out.english_query}")
            except Exception as e:
                print(f"  [error] {item['question'][:50]}: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval eval gate for one language")
    parser.add_argument("--questions", required=True, type=Path, help="JSONL questions file")
    parser.add_argument("--language", required=True, choices=sorted(languages.LANGUAGES))
    parser.add_argument("--k", type=int, default=5, help="Cutoff for hit@k and MRR (default 5)")
    parser.add_argument("--fetch-k", type=int, default=20, help="Matches to fetch per question")
    parser.add_argument("--gate", type=float, default=0.6, help="Minimum hit@k to pass (default 0.6)")
    parser.add_argument("--check-classifier", action="store_true",
                        help="Also spot-check language detection and English translation")
    args = parser.parse_args()

    if not args.questions.exists():
        print(f"Questions file not found: {args.questions}", file=sys.stderr)
        return 2

    lang = languages.get(args.language)
    items = load_questions(args.questions)
    print(f"Evaluating {len(items)} {lang.name} questions against namespace "
          f"'{lang.namespace}' of index '{os.getenv('PINECONE_INDEX', 'website-rag')}'\n")

    report = run_retrieval(items, lang, args.k, args.fetch_k)
    metrics = report["metrics"]
    key = f"hit@{args.k}"

    print("\n=== retrieval")
    print(f"{key}: {metrics[key]:.3f}   MRR@{args.k}: {metrics['mrr']:.3f}")
    for label, values in (
        ("top-1 scores", report["top1_scores"]),
        ("expected-page scores", report["hit_scores"]),
        ("all fetched scores", report["all_scores"]),
    ):
        pcts = ", ".join(f"p{p}={percentile(values, p):.3f}" for p in (10, 25, 50, 75, 90))
        print(f"{label:>22}: {pcts}" if values else f"{label:>22}: none")
    print("Set the language's min_score near the low end of the expected-page scores.")

    if args.check_classifier:
        run_classifier_check(items)

    passed = metrics[key] >= args.gate
    print(f"\nGATE: {'PASS' if passed else 'FAIL'} ({key} {metrics[key]:.3f} vs {args.gate})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
