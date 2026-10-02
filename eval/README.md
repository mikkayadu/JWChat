# Retrieval eval sets

`scripts/eval_retrieval.py` reads one JSONL file per language. Each line is one question:

```json
{"question": "Dɛn nti na Onyankopɔn ma amanehunu kɔ so?", "expected_urls": ["https://www.jw.org/tw/..."]}
```

- `question`: asked the way a user would ask it, in that language.
- `expected_urls`: one or more pages that answer it. A question counts as a hit when any of them comes back in the top k.

## `tw_questions.jsonl` (to be written)

Write about 25 Twi questions after the bounded Twi sample has been crawled into the dev index, using only pages that are actually in that sample. If a question points to a page that was never ingested, the gate measures crawl coverage instead of embedding quality. A fluent Twi speaker should write or review them, and some should be typed without ɛ/ɔ the way people often type on phones.

```bash
PINECONE_INDEX=jwchat-dev python -m ingestion.cli.pipeline \
    --crawl https://www.jw.org/tw/nhomakorabea/ --max-pages 300 --sync

PINECONE_INDEX=jwchat-dev python -m scripts.eval_retrieval \
    --questions eval/tw_questions.jsonl --language tw --check-classifier
```

The gate passes at hit@5 >= 0.6. If it fails, stop before the query-time PR and evaluate `gemini-embedding-2`, which needs a full re-ingest.
