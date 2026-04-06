# RAG Ingestion Pipeline

Ingests PDFs, HTML pages, and videos into a Qdrant vector database
ready to power an AI chatbot.

**You do NOT need to download any files.** Just provide URLs — the pipeline
streams PDFs and videos directly from your website.

## Project Structure

```
JWChat/
├── .env.example
├── .gitignore
├── requirements.txt
├── page_urls.txt
├── pdf_links.txt
├── api/
│   └── app.py
├── static/
│   └── index.html
└── ingestion/
  ├── cli/
  │   └── pipeline.py
  ├── tasks/
  │   └── worker.py
  ├── extractors/
  │   ├── crawler.py
  │   ├── html_scraper.py
  │   ├── pdf_extractor.py
  │   └── video_transcriber.py
  └── utils/
    ├── chunker.py
    └── embedder.py
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

### 2. Configure environment

```bash
cp .env.example .env
# Open .env and add your OPENAI_API_KEY
```

### 3. Start Qdrant (vector database)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

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
- **Qdrant dashboard**: http://localhost:6333/dashboard

---

## Re-indexing

Safe to re-run. Vectors use deterministic IDs based on source URL + chunk index,
so re-ingesting the same URL replaces old vectors rather than creating duplicates.

---

## Next step

Once ingestion is done, build the **Query API** — a FastAPI backend that accepts
user questions, searches Qdrant, and calls Claude to generate answers.
