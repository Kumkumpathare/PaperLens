"""Turn chunks into vectors and search them with cosine similarity."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .ingest import Chunk

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def get_model(name: str = DEFAULT_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


class VectorIndex:
    """Small in-memory index. Fine for a few dozen papers; no extra database needed."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        model = get_model(self.model_name)
        vecs = model.encode([c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False)
        self.vectors = vecs if self.vectors is None else np.vstack([self.vectors, vecs])
        self.chunks.extend(chunks)

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if self.vectors is None:
            return []
        q = get_model(self.model_name).encode([query], normalize_embeddings=True)[0]
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]
