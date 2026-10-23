"""Pick the excerpts worth sending to the LLM."""
from __future__ import annotations

from .embed import VectorIndex
from .ingest import Chunk


def retrieve(index: VectorIndex, question: str, k: int = 4, min_score: float = 0.25) -> list[tuple[Chunk, float]]:
    """Top-k excerpts whose similarity is at least `min_score`.

    If nothing passes the threshold, the question is treated as not answerable
    from the uploaded papers. Tune `min_score` with eval/run_eval.py.
    """
    return [(c, s) for c, s in index.search(question, k) if s >= min_score]
