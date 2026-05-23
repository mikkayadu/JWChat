# JWChat — Jehovah's Witnesses RAG Chatbot

A RAG (Retrieval-Augmented Generation) chatbot for Jehovah's Witnesses content,
ingesting PDFs, HTML pages, and videos into a **Pinecone** vector database
and answering questions using **Gemini**.

**You do NOT need to download any files.** Just provide URLs — the pipeline
streams PDFs and videos directly from your website.

## Features

- **Multi-format ingestion**: PDFs (with OCR for scanned pages), HTML pages, and videos (YouTube/Vimeo/MP4)
- **Automatic URL discovery**: Sitemap parsing and site crawling
- **Parallel processing**: Celery + Redis for handling thousands of files
- **Intent routing**: LangGraph-powered flow that handles greetings, off-topic questions, and domain-specific queries
- **Source citations**: All answers include references to the source material

## RAG System Improvements

The RAG system has been enhanced with state-of-the-art techniques for better retrieval and answer quality. See **[RAG_IMPROVEMENTS.md](RAG_IMPROVEMENTS.md)** for complete details.

### Quick Summary

| Improvement | Status | Impact |
|-------------|--------|--------|
| Recursive Chunking | ✓ Done | +20% context coherence |
| Hybrid Search | ✓ Done | +10-30% recall |
| Reranking | ✓ Done | +15-25% precision |
| Query Transformation (HyDE) | ✓ Done | Better vocabulary matching |
| RAGAS Evaluation | ✓ Done | Quantitative metrics |
| Parent-Child Retrieval | ⚠️ Partial | Better context |

### Quick Configuration

```bash
# Enable in .env
USE_HYBRID_SEARCH=true
USE_RERANKING=true
RETRIEVAL_K=20
RETRIEVAL_MIN_SCORE=0.75
```

For detailed configuration, see [RAG_IMPROVEMENTS.md](RAG_IMPROVEMENTS.md).

## Project Structure

```
JWChat/
├── .env.example
├── .gitignore
├── requirements.txt
├── page_urls.txt
├── pdf_links.txt
├── api/
│   ├── __init__.py
│   ├── app.py                  # FastAPI backend
│   └── services/
│       ├── __init__.py
│       └── chat_service.py     # LangGraph intent routing + RAG
├── static/
│   └── index.html              # Chat UI
└── ingestion/
  ├── __init__.py
  ├── cli/
  │   ├── __init__.py
  │   └── pipeline.py           # CLI entry point
  ├── tasks/
  │   ├── __init__.py
  │   └── worker.py             # Celery tasks
  ├── extractors/
  │   ├── __init__.py           # Shared dataclasses
  │   ├── crawler.py            # Site crawling + sitemap
  │   ├── html_scraper.py       # HTML page extraction
  │   ├── pdf_extractor.py      # PDF extraction (text + OCR)
  │   └── video_transcriber.py  # Video transcription (Whisper)
  └── utils/
      ├── __init__.py
      ├── chunker.py            # Token-aware text chunking
      └── embedder.py           # Gemini embeddings + Pinecone
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium     # For JS-rendered page scraping
```

### 2. Configure environment

```bash
cp .env.example .env
# Open .env and add your OPENAI_API_KEY
```

### 3. Create Pinecone Index (if not exists)

The pipeline auto-creates the Pinecone index on first run. Ensure your `.env`
has the correct `PINECONE_API_KEY`, `PINECONE_INDEX`, `PINECONE_CLOUD`, and
`PINECONE_REGION` values.

### 4. Start Redis + Celery (for parallel processing of thousands of files)

```bash
docker run -p 6379:6379 redis

# In a separate terminal:
celery -A ingestion.tasks.worker worker --loglevel=info --concurrency=4
```

---

## Running the pipeline

### Ingest PDFs from URLs (no downloading needed)

Create `pdf_links.txt` — one URL per line:
```
# Comments and blank lines are ignored
https://yourwebsite.com/docs/annual-report.pdf
https://yourwebsite.com/resources/user-guide.pdf
https://yourwebsite.com/files/whitepaper.pdf
```
```bash
python -m ingestion.cli.pipeline --pdf-urls pdf_links.txt
```

### Ingest videos from URLs

Create `video_links.txt` — supports direct MP4 links AND YouTube/Vimeo:
```
https://yourwebsite.com/videos/intro-lecture.mp4
https://yourwebsite.com/videos/tutorial.mp4
https://youtube.com/watch?v=abc123
https://vimeo.com/123456789
```
```bash
python -m ingestion.cli.pipeline --video-urls video_links.txt --whisper-model base
```

### Ingest HTML pages

Create `page_urls.txt`:
```
https://yourwebsite.com/about
https://yourwebsite.com/faq
https://yourwebsite.com/docs/getting-started
```
```bash
python -m ingestion.cli.pipeline --urls page_urls.txt
```

### Ingest everything at once
```bash
python -m ingestion.cli.pipeline --pdf-urls pdf_links.txt --video-urls video_links.txt --urls page_urls.txt
```

### Test without Celery (no queue, runs one by one)
```bash
python -m ingestion.cli.pipeline --pdf-urls pdf_links.txt --sync
```

---

## Auto-discover URLs from your sitemap

Most websites have a sitemap. Use it to get all URLs automatically:

```bash
# Fetch all URLs from your sitemap
curl https://yourwebsite.com/sitemap.xml | grep -oP '(?<=<loc>)[^<]+' > all_urls.txt

# Split by type
grep '\.pdf'          all_urls.txt > pdf_links.txt
grep -E '\.(mp4|webm)' all_urls.txt > video_links.txt
grep -vE '\.(pdf|mp4|webm)' all_urls.txt > page_urls.txt

# Run everything
python -m ingestion.cli.pipeline --pdf-urls pdf_links.txt --video-urls video_links.txt --urls page_urls.txt
```

---

## Local files also work

```bash
python -m ingestion.cli.pipeline --pdfs ./my_documents
python -m ingestion.cli.pipeline --videos ./my_videos
```

---

## Monitoring

- **Celery Flower** (task dashboard): `pip install flower && celery -A worker flower`
  Then open http://localhost:5555
- **Pinecone console**: https://app.pinecone.io

---

## Re-indexing

Safe to re-run. Vectors use deterministic IDs based on source URL + chunk index,
so re-ingesting the same URL replaces old vectors rather than creating duplicates.

---

## Running the Chatbot

After ingestion is complete, start the API server:

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000 in your browser to chat.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat UI |
| `/health` | GET | Health check |
| `/api/chat` | POST | Send a question |

### Example Chat Request

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the Bible say about love?", "top_k": 5}'
```

### Intent Routing

The chatbot uses LangGraph to classify queries:

- **Greetings** → Friendly welcome message
- **Off-topic** → Polite redirect to JW content
- **In-domain** → RAG retrieval + Gemini answer with citations
