"""Tests for the pure logic of citation verification (no LLM calls)."""
from __future__ import annotations

import pytest

from measurable_rag.generation.verify import (
    CitationVerdict,
    citation_precision,
    faithfulness,
    split_claims,
)


def test_split_claims_extracts_text_and_citations():
    answer = "Fat raises risk [1]. Obesity links to disease [2][3]. This has no citation."
    claims = split_claims(answer)
    assert claims == [
        ("Fat raises risk.", [1]),
        ("Obesity links to disease.", [2, 3]),
        ("This has no citation.", []),
    ]


def test_split_claims_drops_citation_only_sentences():
    # A trailing bare "[1][3]." is not a claim — no letters once citations strip.
    answer = "Supported statement [1]. [1][3]."
    assert split_claims(answer) == [("Supported statement.", [1])]


def _v(claim: str, supported: bool) -> CitationVerdict:
    return CitationVerdict(claim=claim, source_num=1, doc_id="d", supported=supported)


def test_citation_precision_counts_supported_links():
    verdicts = [_v("a", True), _v("b", True), _v("c", False)]
    assert citation_precision(verdicts) == pytest.approx(2 / 3)


def test_citation_precision_is_none_without_citations():
    assert citation_precision([]) is None


def test_faithfulness_is_per_claim_not_per_citation():
    # One claim cited two sources: one supports, one doesn't -> claim is faithful.
    verdicts = [
        CitationVerdict("claim X", 1, "d1", True),
        CitationVerdict("claim X", 2, "d2", False),
        CitationVerdict("claim Y", 1, "d1", False),
    ]
    # claim X supported, claim Y not -> 1 of 2 claims faithful.
    assert faithfulness(verdicts) == pytest.approx(0.5)
    # ...but citation precision is per-link: 1 of 3 links supported.
    assert citation_precision(verdicts) == pytest.approx(1 / 3)
