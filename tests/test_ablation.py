"""Spec for run_ablation (your code in eval/ablation.py).

FAILS until you implement run_ablation. Uses fake config functions and a tiny
answer key so it tests your ORCHESTRATION (chunk->doc mapping + scoring + naming)
without needing any models or the real index.

    python -m pytest tests/test_ablation.py -q
"""
from __future__ import annotations

import pytest

from measurable_rag.data.models import Chunk
from measurable_rag.eval.ablation import run_ablation


def chunk(doc_id: str, suffix: str = "a") -> Chunk:
    """Minimal chunk for a given document id."""
    return Chunk(chunk_id=f"{doc_id}-{suffix}", doc_id=doc_id, text="t", start=0, end=1)


# Two queries (keyed by id); the config functions receive the query TEXT.
QUERIES = {"q1": "alpha", "q2": "beta"}
# Answer key: q1's relevant doc is A, q2's is B.
QRELS = {"q1": {"A": 1}, "q2": {"B": 1}}


def oracle(query_text: str) -> list[Chunk]:
    """Always returns the relevant doc first."""
    return [chunk("A")] if query_text == "alpha" else [chunk("B")]


def wrong(query_text: str) -> list[Chunk]:
    """Never returns a relevant doc."""
    return [chunk("Z"), chunk("Y")]


def dupes(query_text: str) -> list[Chunk]:
    """Relevant doc appears via two chunks before a distractor — tests that the
    chunk->doc collapse keeps the doc at the rank of its FIRST (best) chunk."""
    if query_text == "alpha":
        return [chunk("A", "c1"), chunk("A", "c2"), chunk("Z")]
    return [chunk("B", "c1"), chunk("B", "c2"), chunk("Z")]


def test_returns_one_report_per_config():
    reports = run_ablation({"oracle": oracle, "wrong": wrong}, QUERIES, QRELS)
    assert set(reports) == {"oracle", "wrong"}


def test_oracle_scores_perfect():
    reports = run_ablation({"oracle": oracle}, QUERIES, QRELS)
    r = reports["oracle"]
    assert r["Recall@1"] == pytest.approx(1.0)
    assert r["MRR"] == pytest.approx(1.0)
    assert r["nDCG@1"] == pytest.approx(1.0)


def test_wrong_scores_zero():
    reports = run_ablation({"wrong": wrong}, QUERIES, QRELS)
    r = reports["wrong"]
    assert r["Recall@1"] == pytest.approx(0.0)
    assert r["MRR"] == pytest.approx(0.0)


def test_chunks_collapse_to_documents():
    # Relevant doc reached via 2 chunks then a distractor -> after collapsing to
    # documents the relevant doc is still rank 1, so Recall@1 == 1.0.
    reports = run_ablation({"dupes": dupes}, QUERIES, QRELS)
    assert reports["dupes"]["Recall@1"] == pytest.approx(1.0)
