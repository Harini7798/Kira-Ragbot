"""Cross-encoder reranking: a slow-but-accurate second pass over the top candidates.

The dense and BM25 retrievers encode the query and each chunk *separately*, then
compare — fast (you can pre-index the whole corpus), but the query never "sees"
the chunk during encoding.

A CROSS-encoder feeds ``(query, chunk)`` through one transformer together and
outputs a single relevance score, so it can model fine-grained interactions
("does THIS passage actually answer THIS question?"). It ranks better, but must
run the model once per (query, chunk) pair — far too slow for 15k chunks. Hence
the standard recipe we follow: retrieve a cheap top-N with dense+BM25, then
rerank only those N.
"""
from __future__ import annotations

import numpy as np

from .. import config
from ..data.models import Chunk


class Reranker:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or config.RERANKER_MODEL
        self.model = CrossEncoder(self.model_name, device=device)
        self.device = str(getattr(self.model, "device", device or "cpu"))

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Re-order ``chunks`` by cross-encoder relevance to ``query`` (best first)."""
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        scores = self.model.predict(pairs)
        order = np.argsort(scores)[::-1]  # highest score first
        return [chunks[i] for i in order]
