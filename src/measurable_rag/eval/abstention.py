"""Measure abstention: false-answer rate vs. over-refusal rate over a threshold sweep.

This produces the project's standout artifact. Each eval question is run once
through the system, yielding a small record: its label (answerable /
unanswerable), the grounding score of the generated answer, and whether the
generator flagged the sources as insufficient. We then sweep the grounding
threshold and, at each value, compute the two rates that are in tension:

  * false_answer_rate — of UNANSWERABLE questions, the fraction the system still
    answered (confabulation). Want LOW.
  * over_refusal_rate — of ANSWERABLE questions, the fraction the system refused.
    Want LOW.

Because the records are precomputed, the sweep is just re-thresholding — cheap,
so we can plot a fine-grained curve without re-running the LLM.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AbstentionRecord:
    qid: str
    label: str            # "answerable" or "unanswerable"
    grounding: float      # grounding_score of the generated answer
    insufficient: bool    # generator said the sources don't answer it


def _abstains(record: AbstentionRecord, threshold: float) -> bool:
    return record.insufficient or record.grounding < threshold


def sweep(
    records: list[AbstentionRecord], thresholds: list[float]
) -> list[dict[str, float]]:
    """For each threshold, the false-answer and over-refusal rates."""
    answerable = [r for r in records if r.label == "answerable"]
    unanswerable = [r for r in records if r.label == "unanswerable"]

    rows: list[dict[str, float]] = []
    for t in thresholds:
        far = (
            sum(1 for r in unanswerable if not _abstains(r, t)) / len(unanswerable)
            if unanswerable
            else 0.0
        )
        orr = (
            sum(1 for r in answerable if _abstains(r, t)) / len(answerable)
            if answerable
            else 0.0
        )
        rows.append(
            {"threshold": t, "false_answer_rate": far, "over_refusal_rate": orr}
        )
    return rows


def best_threshold(rows: list[dict[str, float]]) -> dict[str, float]:
    """The sweep row minimizing the sum of the two error rates — a simple,
    defensible 'balanced' operating point to quote alongside the full curve."""
    return min(rows, key=lambda r: r["false_answer_rate"] + r["over_refusal_rate"])
