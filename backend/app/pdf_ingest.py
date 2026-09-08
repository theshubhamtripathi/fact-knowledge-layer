"""Per-page text extraction from any PDF."""
from dataclasses import dataclass

import pdfplumber


@dataclass
class Page:
    page_number: int  # 1-indexed
    text: str


def extract_pages(pdf_path: str) -> list[Page]:
    pages: list[Page] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append(Page(page_number=i, text=text))
    return pages


def window_pages(pages: list[Page], window_size: int = 3, overlap: int = 1) -> list[list[Page]]:
    """Group pages into small overlapping windows so large PDFs are processed
    in bounded chunks rather than one giant LLM call."""
    if window_size <= overlap:
        raise ValueError("window_size must be greater than overlap")
    windows: list[list[Page]] = []
    step = window_size - overlap
    i = 0
    while i < len(pages):
        windows.append(pages[i : i + window_size])
        if i + window_size >= len(pages):
            break
        i += step
    return windows
