"""BM25 sparse (keyword) retrieval over the same chunks as the dense index.

Dense retrieval matches by *meaning* but can miss when a query hinges on an exact
token the embedder smooths over — a gene name, an acronym, a rare term. BM25 is
the classic answer: it scores a chunk by how often the query's words appear in
it, weighted so rare words count more and long chunks don't win just by length.
It knows nothing about meaning, which is exactly why it complements the dense
retriever — they fail on different queries.

"Sparse" because each chunk is represented by counts over the vocabulary: a
huge vector that's almost all zeros (only the words actually present are
non-zero) — the opposite of the dense embedding's small, fully-filled vector.

We build the index over the dense index's exact chunk list so the two retrievers
are searching identical content — a prerequisite for a fair ablation.
"""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from ..data.models import Chunk

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Deliberately simple — BM25 already down-weights
    common words via its term statistics, so we skip stopword lists for now."""
    return _TOKEN.findall(text.lower())


class SparseIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        # BM25Okapi precomputes document frequencies / lengths up front; queries
        # are then cheap. Building over ~15k short chunks takes a second or two.
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """Top-k (chunk, BM25 score) for the query, most relevant first."""
        scores = self.bm25.get_scores(tokenize(query))  # one score per chunk
        if k >= len(scores):
            order = np.argsort(scores)[::-1]
        else:
            # argpartition grabs the top-k unordered in O(n), then we sort just
            # those k — cheaper than fully sorting all ~15k scores.
            top = np.argpartition(scores, -k)[-k:]
            order = top[np.argsort(scores[top])[::-1]]
        return [(self.chunks[i], float(scores[i])) for i in order]
