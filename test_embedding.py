"""Smoke test PDF processing and sentence embedding generation."""

from pathlib import Path

from src.embedding import EXPECTED_EMBEDDING_DIMENSIONS, EmbeddingModel
from src.pdf_processor import chunk_text, extract_text_from_pdf


PROJECT_ROOT = Path(__file__).resolve().parent
PDF_PATH = PROJECT_ROOT / "data" / "documents" / "nist_incident_response.pdf"


def main() -> None:
    pages = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_text(pages, chunk_size=500, overlap=100)
    if not chunks:
        raise RuntimeError("PDF text extraction produced no chunks.")

    embeddings = EmbeddingModel().generate_embeddings(
        [chunk["text"] for chunk in chunks]
    )
    expected_shape = (len(chunks), EXPECTED_EMBEDDING_DIMENSIONS)
    if embeddings.shape != expected_shape:
        raise AssertionError(
            f"Expected embedding shape {expected_shape}, got {embeddings.shape}."
        )

    print(f"PDF pages: {len(pages)}")
    print(f"Text chunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print("Embedding check passed.")


if __name__ == "__main__":
    main()
