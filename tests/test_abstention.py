"""Tests for the pure logic of abstention and the trade-off sweep (no LLM calls)."""
from __future__ import annotations

import pytest

from measurable_rag.eval.abstention import AbstentionRecord, best_threshold, sweep
from measurable_rag.generation.abstain import grounding_score, should_abstain
from measurable_rag.generation.verify import CitationVerdict


def _v(claim: str, supported: bool) -> CitationVerdict:
    return CitationVerdict(claim=claim, source_num=1, doc_id="d", supported=supported)


def test_grounding_score_is_fraction_of_supported_claims():
    verdicts = [_v("a", True), _v("b", False), _v("c", True), _v("c", False)]
    # claim c counts once and is supported (any supporting citation) -> a,c of a,b,c
    assert grounding_score(verdicts) == pytest.approx(2 / 3)


def test_grounding_score_zero_without_citations():
    assert grounding_score([]) == 0.0


def test_should_abstain_on_insufficient_regardless_of_threshold():
    assert should_abstain("INSUFFICIENT: ...", [_v("a", True)], threshold=0.0) is True


def test_should_abstain_respects_threshold():
    verdicts = [_v("a", True), _v("b", False)]  # grounding = 0.5
    assert should_abstain("answer", verdicts, threshold=0.4) is False
    assert should_abstain("answer", verdicts, threshold=0.6) is True


# --- the trade-off sweep ----------------------------------------------------
RECORDS = [
    AbstentionRecord("a1", "answerable", grounding=0.9, insufficient=False),
    AbstentionRecord("a2", "answerable", grounding=0.3, insufficient=False),
    AbstentionRecord("u1", "unanswerable", grounding=0.2, insufficient=False),
    AbstentionRecord("u2", "unanswerable", grounding=0.0, insufficient=True),
]


def test_sweep_low_threshold_answers_everything():
    # threshold 0.0: only the 'insufficient' one abstains.
    row = sweep(RECORDS, [0.0])[0]
    assert row["false_answer_rate"] == pytest.approx(0.5)   # u1 answered, u2 refused
    assert row["over_refusal_rate"] == pytest.approx(0.0)   # both answerable answered


def test_sweep_mid_threshold_trades_off():
    # threshold 0.5: a2 (0.3) over-refused; u1 (0.2) correctly refused.
    row = sweep(RECORDS, [0.5])[0]
    assert row["false_answer_rate"] == pytest.approx(0.0)
    assert row["over_refusal_rate"] == pytest.approx(0.5)


def test_best_threshold_minimizes_total_error():
    rows = sweep(RECORDS, [0.0, 0.25, 0.5, 1.0])
    best = best_threshold(rows)
    # threshold 0.25: u1(0.2)&u2 refused -> far 0; a1 answered, a2(0.3) answered -> orr 0.
    assert best["threshold"] == pytest.approx(0.25)
    assert best["false_answer_rate"] == pytest.approx(0.0)
    assert best["over_refusal_rate"] == pytest.approx(0.0)
