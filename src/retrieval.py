"""Small FAISS helpers for indexing and searching PDF text chunks."""

import json
from pathlib import Path

import faiss
import numpy as np


def create_faiss_index(embeddings):
    """Build a cosine-similarity FAISS index from embedding vectors."""
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty 2D array.")

    vectors = vectors.copy()
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_vectorstore(index, chunks, directory):
    """Save the FAISS index and its matching chunk metadata locally."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(directory / "chunks.faiss"))
    with (directory / "chunks.json").open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)


def load_vectorstore(directory):
    """Load a previously saved FAISS index and matching chunks."""
    directory = Path(directory)
    index_path = directory / "chunks.faiss"
    chunks_path = directory / "chunks.json"
    if not index_path.is_file() or not chunks_path.is_file():
        raise FileNotFoundError(
            f"No saved index found in {directory}. Process a PDF first."
        )

    index = faiss.read_index(str(index_path))
    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)
    if index.ntotal != len(chunks):
        raise ValueError("The FAISS index and saved chunk list do not match.")
    return index, chunks


def retrieve_chunks(question, index, chunks, embedding_model, top_k=3):
    """Return the most similar chunks with their source and page metadata."""
    if not question or not question.strip():
        raise ValueError("Enter a question before searching.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if index.ntotal == 0:
        return []

    query_vector = np.asarray(
        embedding_model.generate_embeddings([question]), dtype=np.float32
    )
    if query_vector.ndim != 2 or query_vector.shape[1] != index.d:
        raise ValueError(
            f"Question embedding dimensions do not match FAISS index ({index.d})."
        )
    faiss.normalize_L2(query_vector)
    scores, positions = index.search(query_vector, min(top_k, index.ntotal))

    results = []
    for score, position in zip(scores[0], positions[0]):
        if position >= 0:
            result = dict(chunks[int(position)])
            result["score"] = float(score)
            results.append(result)
    return results
