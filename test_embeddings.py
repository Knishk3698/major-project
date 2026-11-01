"""Smoke test PDF extraction, chunking, and 384-dimensional embeddings."""

from pathlib import Path

from src.embedding import EXPECTED_EMBEDDING_DIMENSIONS, EmbeddingModel
from src.pdf_processor import chunk_text, extract_text_from_pdf


PROJECT_ROOT = Path(__file__).resolve().parent
PDF_PATH = PROJECT_ROOT / "data" / "documents" / "nist_incident_response.pdf"


def main() -> None:
    print(f"Extracting text from: {PDF_PATH}")
    pages = extract_text_from_pdf(PDF_PATH)
    for page in pages:
        page["source"] = str(PDF_PATH)

    chunks = chunk_text(pages, chunk_size=500, overlap=100)
    if not chunks:
        raise RuntimeError("PDF text extraction produced no chunks.")
    print(f"Number of chunks: {len(chunks)}")

    texts = [chunk["text"] for chunk in chunks]
    model = EmbeddingModel()
    embeddings = model.generate_embeddings(texts)

    print(f"Number of embeddings: {embeddings.shape[0]}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    if embeddings.shape != (len(chunks), EXPECTED_EMBEDDING_DIMENSIONS):
        raise AssertionError(
            f"Expected shape ({len(chunks)}, {EXPECTED_EMBEDDING_DIMENSIONS}), "
            f"got {embeddings.shape}."
        )
    print("Confirmed: every chunk has a 384-dimensional embedding.")


if __name__ == "__main__":
    main()
