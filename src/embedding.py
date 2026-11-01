"""Sentence Transformer wrapper for project text embeddings."""

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_EMBEDDING_DIMENSIONS = 384


class EmbeddingModel:
    """Encode text into dense NumPy vectors using a pretrained model."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("Embedding model loaded successfully.")

    def generate_embeddings(self, texts: Sequence[str]) -> np.ndarray:
        """Return one 384-dimensional NumPy embedding per non-empty text."""
        if isinstance(texts, (str, bytes)) or not texts:
            raise ValueError("Provide a non-empty list of text chunks.")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Every text chunk must be a non-empty string.")

        vectors = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        vectors = np.asarray(vectors)
        if vectors.ndim != 2 or vectors.shape[1] != EXPECTED_EMBEDDING_DIMENSIONS:
            raise ValueError(
                "Unexpected embedding shape: "
                f"{vectors.shape}; expected (number_of_texts, "
                f"{EXPECTED_EMBEDDING_DIMENSIONS})."
            )
        return vectors
