"""Spec for the retrieval metrics you implement in eval/metrics.py.

These FAIL until you implement the functions (the stubs raise NotImplementedError
— that's intentional). Each case is worked out by hand in the comments so you can
check your arithmetic. Run just these with:

    python -m pytest tests/test_metrics.py -q
"""
from __future__ import annotations

import pytest

from measurable_rag.eval.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Ranked docs (best first); the 2 relevant ones sit at ranks 2 and 4.
R = ["a", "b", "c", "d", "e"]
REL = {"b", "d"}


# --- recall@k: fraction of all relevant docs found in the top k -------------
def test_recall_at_k():
    assert recall_at_k(R, REL, 1) == pytest.approx(0.0)    # top1=[a]: 0 of 2
    assert recall_at_k(R, REL, 2) == pytest.approx(0.5)    # top2=[a,b]: 1 of 2
    assert recall_at_k(R, REL, 3) == pytest.approx(0.5)    # top3 adds c: still 1 of 2
    assert recall_at_k(R, REL, 5) == pytest.approx(1.0)    # both found


def test_recall_with_k_beyond_list_length():
    # Only 2 docs retrieved, 3 relevant exist -> at most 2/3 recoverable.
    assert recall_at_k(["a", "b"], {"a", "b", "z"}, 5) == pytest.approx(2 / 3)


# --- precision@k: fraction of the top k that are relevant -------------------
def test_precision_at_k():
    assert precision_at_k(R, REL, 1) == pytest.approx(0.0)   # 0 of 1
    assert precision_at_k(R, REL, 2) == pytest.approx(0.5)   # 1 of 2
    assert precision_at_k(R, REL, 5) == pytest.approx(0.4)   # 2 of 5


# --- reciprocal rank: 1 / rank of the first relevant doc --------------------
def test_reciprocal_rank():
    assert reciprocal_rank(R, REL) == pytest.approx(0.5)               # first hit at rank 2
    assert reciprocal_rank(["b", "a"], REL) == pytest.approx(1.0)      # hit at rank 1
    assert reciprocal_rank(["x", "y", "z"], REL) == pytest.approx(0.0) # no hit


# --- nDCG@k: rewards ranking relevant docs higher ---------------------------
def test_ndcg_rewards_higher_ranks():
    # gains: rank2 (b)=1, rank4 (d)=1; discount = 1/log2(rank+1)
    # DCG  = 1/log2(3) + 1/log2(5) = 0.63093 + 0.43068 = 1.06161
    # IDCG = ideal puts both at ranks 1,2: 1/log2(2) + 1/log2(3) = 1.0 + 0.63093 = 1.63093
    # nDCG@5 = 1.06161 / 1.63093 = 0.65092
    assert ndcg_at_k(R, REL, 5) == pytest.approx(0.65092, abs=1e-4)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)


def test_ndcg_idcg_is_capped_at_k():
    # k=1, one slot. Best possible is one relevant doc at rank 1 -> IDCG = 1.0.
    # retrieved has b (relevant) at rank 1 -> DCG@1 = 1.0 -> nDCG@1 = 1.0,
    # even though a second relevant doc (d) exists but can't fit in 1 slot.
    assert ndcg_at_k(["b", "a", "c"], {"b", "d"}, 1) == pytest.approx(1.0)


def test_ndcg_discount_at_lower_rank():
    # Single relevant doc sitting at rank 3: DCG = 1/log2(4) = 0.5; IDCG = 1.0.
    assert ndcg_at_k(["a", "b", "c"], {"c"}, 3) == pytest.approx(0.5)


# --- all metrics agree there's nothing to find ------------------------------
def test_no_relevant_retrieved_is_zero():
    none = ["x", "y", "z"]
    assert recall_at_k(none, {"w"}, 3) == pytest.approx(0.0)
    assert precision_at_k(none, {"w"}, 3) == pytest.approx(0.0)
    assert reciprocal_rank(none, {"w"}) == pytest.approx(0.0)
    assert ndcg_at_k(none, {"w"}, 3) == pytest.approx(0.0)
