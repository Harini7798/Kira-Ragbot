"""Compose the retrievers into the three configurations the ablation compares.

Each method takes a query string and returns a ranked list of ``Chunk`` (best
first). They all return the same depth (``TOP_K`` chunks) so the ablation scores
them on equal footing — only the retrieval *method* differs.

  * ``dense``            — meaning-based only (the M2 baseline).
  * ``hybrid``           — dense + BM25 fused with RRF.
  * ``hybrid_reranked``  — fuse a candidate pool, then cross-encoder rerank it.
"""
from __future__ import annotations

from .. import config
from ..data.models import Chunk
from .fusion import reciprocal_rank_fusion


class HybridRetriever:
    def __init__(self, dense_index, sparse_index, embedder, reranker=None):
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.embedder = embedder
        self.reranker = reranker

    def dense(self, query: str, k: int | None = None) -> list[Chunk]:
        k = k or config.TOP_K
        q_emb = self.embedder.encode_queries([query])[0]
        return [c for c, _ in self.dense_index.search(q_emb, k)]

    def sparse(self, query: str, k: int | None = None) -> list[Chunk]:
        k = k or config.TOP_K
        return [c for c, _ in self.sparse_index.search(query, k)]

    def hybrid(self, query: str, k: int | None = None) -> list[Chunk]:
        k = k or config.TOP_K
        dense_hits = self.dense(query, k)
        sparse_hits = self.sparse(query, k)
        return reciprocal_rank_fusion([dense_hits, sparse_hits])[:k]

    def hybrid_reranked(
        self, query: str, k: int | None = None, pool: int | None = None
    ) -> list[Chunk]:
        k = k or config.TOP_K
        pool = pool or config.RERANK_POOL
        if self.reranker is None:
            raise ValueError("HybridRetriever was built without a reranker")
        candidates = self.hybrid(query, pool)  # fuse a pool, then rerank it
        return self.reranker.rerank(query, candidates)[:k]
