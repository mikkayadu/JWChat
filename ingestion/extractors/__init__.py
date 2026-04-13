"""
Extractors — shared base classes
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExtractedPage:
    """
    Base class for extracted content from any source.
    
    Common fields:
        text:        Extracted text content
        source_path: Where this came from (URL or file path)
        metadata:    Additional info (file_type, page_number, etc.)
    
    Source-type specific fields (optional):
        url:         For HTML pages, the URL
        title:       For HTML pages, the page title
        page_number: For PDFs, the page number (1-indexed)
        is_ocr:      For PDFs, whether text came from OCR
    """
    text: str
    source_path: str = ""
    metadata: dict = field(default_factory=dict)
    
    # HTML-specific fields
    url: Optional[str] = None
    title: Optional[str] = None
    
    # PDF-specific fields
    page_number: Optional[int] = None
    is_ocr: bool = False


# Re-export for convenience
__all__ = ["ExtractedPage"]
