"""Retrieval metrics — the measurement core of the project.

Each function scores ONE query:
  * ``retrieved`` — the ranked list of document ids the retriever returned, best
    first, already de-duplicated (rank 1 = retrieved[0]).
  * ``relevant``  — the set of document ids the answer key marks relevant.
The harness averages each metric over all queries (e.g. MRR is just the mean of
``reciprocal_rank`` across queries).

Conventions:
  * Ranks are 1-indexed: the first retrieved doc is at rank 1.
  * Relevance is binary: a doc is relevant iff it is in ``relevant``.
  * nDCG uses gain = 1 for a relevant doc / 0 otherwise, a discount of
    1 / log2(rank + 1), and normalizes by the IDCG — the DCG of the ideal
    ranking ("all relevant docs first"), limited to k slots.
  * If ``relevant`` is empty we return 0.0 rather than dividing by zero.
"""
from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Of all the relevant docs, what fraction appear in the top k?

    recall@k = (# relevant docs found in retrieved[:k]) / (total # relevant docs)

    The metric that matters most: if the relevant doc isn't in the top k, the
    generator can't possibly use it.
    """
    if not relevant:
        return 0.0
    # Set intersection counts how many of the top-k are relevant. Using a set
    # also means a duplicate doc id in the top-k can't be double-counted.
    hits = len(set(retrieved[:k]) & relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Of the top k retrieved docs, what fraction are relevant?

    precision@k = (# relevant docs found in retrieved[:k]) / k
    """
    if k <= 0:
        return 0.0
    # Same numerator as recall, but normalized by k (the size of the result set
    # we're judging) instead of by the number of relevant docs.
    hits = len(set(retrieved[:k]) & relevant)
    return hits / k


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1 / (rank of the first relevant doc); 0.0 if none was retrieved.

    The mean of this over all queries is MRR — how high, on average, the first
    correct result lands.
    """
    # Walk the ranking top-down; the first relevant hit decides the score.
    for rank, doc_id in enumerate(retrieved, start=1):  # rank is 1-indexed
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _dcg(relevances: list[int]) -> float:
    """Discounted Cumulative Gain of a ranked list of binary gains (1/0).

    Position i (1-indexed) contributes gain / log2(i + 1). The discount means a
    relevant doc earns less the further down it sits — rank 1 gets 1/log2(2)=1,
    rank 2 gets 1/log2(3)≈0.63, and so on.
    """
    return sum(gain / math.log2(i + 1) for i, gain in enumerate(relevances, start=1))


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k.

    nDCG = DCG(actual top-k) / IDCG(ideal top-k). The ideal ranking puts every
    relevant doc as high as possible; only ``min(k, #relevant)`` of them can fit
    in k slots, so IDCG is capped there. The ratio lands in [0, 1]: 1.0 means the
    relevant docs are ranked exactly as well as possible.
    """
    if not relevant:
        return 0.0

    # Actual: 1 where the retrieved doc at this rank is relevant, else 0.
    actual_gains = [1 if doc_id in relevant else 0 for doc_id in retrieved[:k]]
    dcg = _dcg(actual_gains)

    # Ideal: as many 1s as relevant docs can fit in k slots, all at the top.
    ideal_gains = [1] * min(k, len(relevant))
    idcg = _dcg(ideal_gains)

    return dcg / idcg if idcg > 0 else 0.0
