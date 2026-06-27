"""Reciprocal Rank Fusion (RRF): merge several rankings into one.

The problem: the dense retriever scores chunks by cosine similarity (roughly
0–1) and BM25 by an open-ended positive number. You can't just add those — the
scales are incomparable, and normalizing them is fiddly and fragile.

RRF sidesteps it by ignoring the scores entirely and using only *rank position*.
Each chunk earns 1 / (k + rank) from every list it appears in (rank is
1-indexed), and we sum those contributions. A chunk ranked highly by BOTH
retrievers rises to the top; a chunk only one retriever found still gets a
boost. The constant k (default 60) softens the gap between ranks 1 and 2 so a
single list can't completely dominate.
"""
from __future__ import annotations

from .. import config
from ..data.models import Chunk


def reciprocal_rank_fusion(
    rankings: list[list[Chunk]], k: int | None = None
) -> list[Chunk]:
    """Fuse multiple ranked chunk lists into one ranked list (best first).

    Chunks are identified by ``chunk_id``, so the same chunk found by different
    retrievers is correctly recognized as one item and its contributions add up.
    """
    k = config.RRF_K if k is None else k
    scores: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_by_id[chunk.chunk_id] = chunk
    ordered_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [chunk_by_id[cid] for cid in ordered_ids]
