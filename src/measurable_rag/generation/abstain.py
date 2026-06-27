"""Abstention — answer only when the answer is grounded; otherwise refuse.

The brief is explicit that a bare retrieval-score threshold is weak. Instead we
use a GROUNDING check: generate an answer, verify its claims against the
retrieved sources (verify.py), and refuse if too few claims are grounded — or if
the generator itself flagged the sources as insufficient.

The grounding threshold is tunable. Sweeping it (eval/abstention.py) traces the
tension between two error types:
  * confabulating on questions the corpus can't answer (false-answer rate), and
  * refusing questions it actually could answer (over-refusal rate).
"""
from __future__ import annotations

from .generator import INSUFFICIENT
from .verify import CitationVerdict


def grounding_score(verdicts: list[CitationVerdict]) -> float:
    """Fraction of distinct cited claims that have >=1 supported citation.

    0.0 if the answer cited nothing — an uncited answer is, by definition, not
    grounded in the sources.
    """
    by_claim: dict[str, bool] = {}
    for v in verdicts:
        by_claim[v.claim] = by_claim.get(v.claim, False) or v.supported
    if not by_claim:
        return 0.0
    return sum(by_claim.values()) / len(by_claim)


def should_abstain(
    answer: str, verdicts: list[CitationVerdict], threshold: float
) -> bool:
    """Refuse if the model flagged insufficiency, or grounding is below threshold."""
    if answer.strip().startswith("INSUFFICIENT"):
        return True
    return grounding_score(verdicts) < threshold
