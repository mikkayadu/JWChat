"""
PDF Ingestion
────────────
Handles both text-based and scanned (OCR) PDFs.
Accepts local file paths OR remote URLs — no downloading needed.
Uses PyMuPDF for fast text extraction, falls back to
Tesseract OCR when a page has no selectable text.
"""

import io
import fitz  # PyMuPDF
from pathlib import Path
from loguru import logger

from ingestion.extractors import ExtractedPage


def extract_pdf_from_url(url: str) -> list[ExtractedPage]:
    """
    Stream a PDF directly from a URL and extract its text.
    Nothing is saved to disk.
    """
    import httpx

    logger.info(f"Streaming PDF from URL: {url}")
    try:
        resp = httpx.get(url, timeout=60, follow_redirects=True,
                         headers={"User-Agent": "RAG-Bot/1.0"})
        resp.raise_for_status()
        pdf_bytes = resp.content
    except Exception as e:
        logger.error(f"Failed to download PDF from {url}: {e}")
        return []

    # Open from bytes — no temp file needed
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        logger.error(f"Failed to parse PDF from {url}: {e}")
        return []

    file_name = url.split("/")[-1].split("?")[0] or "remote.pdf"
    return _extract_pages(doc, source=url, file_name=file_name)


def extract_pdf(file_path: str | Path) -> list[ExtractedPage]:
    """
    Extract text from a local PDF file.
    Also accepts a URL string — delegates to extract_pdf_from_url().
    """
    # Transparently handle URLs passed to this function too
    if str(file_path).startswith("http://") or str(file_path).startswith("https://"):
        return extract_pdf_from_url(str(file_path))

    path = Path(file_path)
    logger.info(f"Opening PDF: {path.name}")

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        logger.error(f"Failed to open {path}: {e}")
        return []

    return _extract_pages(doc, source=str(path), file_name=path.name)


def _extract_pages(doc: fitz.Document, source: str, file_name: str) -> list[ExtractedPage]:
    """Shared extraction logic for both local and remote PDFs."""
    pages: list[ExtractedPage] = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()

        is_ocr = False
        if not text:
            text = _ocr_page(page)
            is_ocr = True

        if not text:
            logger.warning(f"  Page {page_num}: no text extracted, skipping")
            continue

        pages.append(ExtractedPage(
            page_number=page_num,
            text=text,
            source_path=source,
            is_ocr=is_ocr,
            metadata={
                "file_name": file_name,
                "file_type": "pdf",
                "source_url": source if source.startswith("http") else "",
                "total_pages": len(doc),
            }
        ))
        logger.debug(f"  Page {page_num}: {len(text)} chars {'(OCR)' if is_ocr else ''}")

    doc.close()
    logger.success(f"PDF done: {len(pages)} pages from '{file_name}'")
    return pages


def _ocr_page(page: fitz.Page) -> str:
    """Render page to image and run Tesseract OCR on it."""
    try:
        import pytesseract
        from PIL import Image
        import io

        # Render at 2x resolution for better OCR accuracy
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        return pytesseract.image_to_string(img).strip()
    except ImportError:
        logger.warning("pytesseract not installed — skipping OCR for this page")
        return ""
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return ""
