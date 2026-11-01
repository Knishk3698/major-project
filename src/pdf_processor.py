
import os
from pathlib import Path

import pymupdf


def extract_text_from_pdf(pdf_path, source_name=None):
    """
    Extract text from every page of a PDF.

    Returns a list of dictionaries containing:
    page number and extracted text.
    """

    if isinstance(pdf_path, (bytes, bytearray)):
        pdf_bytes = pdf_path
    else:
        pdf_path = Path(pdf_path)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        pdf_bytes = None

    try:
        if pdf_bytes is not None:
            document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        else:
            document = pymupdf.open(pdf_path)
        with document:
            pages = []
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append({"page_number": page_number, "text": text})
    except Exception as error:
        raise ValueError(f"Could not read PDF: {error}")

    if not pages:
        raise ValueError("No extractable text found in the PDF.")

    if source_name:
        for page in pages:
            page["source"] = os.path.basename(source_name)
    return pages


def chunk_text(pages, chunk_size=500, overlap=100):
    """
    Divide extracted PDF text into smaller chunks.

    chunk_size: Maximum approximate number of words per chunk.
    overlap: Number of words shared between consecutive chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    chunks = []

    for page in pages:
        words = page["text"].split()

        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))

            chunk_words = words[start:end]
            chunk_content = " ".join(chunk_words)

            if chunk_content.strip():
                chunks.append({
                    "text": chunk_content,
                    "source": os.path.basename(page.get("source", "")),
                    "page_number": page["page_number"]
                })

            if end == len(words):
                break

            start = end - overlap

    return chunks
