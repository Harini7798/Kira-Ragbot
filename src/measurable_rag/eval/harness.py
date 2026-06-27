"""Eval harness: run retrieval over every query, then score it with the metrics.

Deliberately split into two halves:

* ``build_rankings`` — the slow part: embed each query, search the index, and
  collapse the ranked *chunks* into ranked unique *document ids* (the unit the
  answer key uses). A document's rank is the position of its best-scoring chunk.
* ``evaluate`` — the fast part: feed those rankings + the answer key to your
  metric functions and average over queries.

Keeping them separate (and caching the rankings to disk) means that once
retrieval has run once, you can edit metrics.py and re-score instantly without
re-embedding 300 queries.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import metrics


def dedupe_keep_order(items: list[str]) -> list[str]:
    """Drop later duplicates, keep first occurrence — so a document keeps the
    rank of its highest-scoring chunk."""
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_rankings(index, embedder, queries: dict[str, str], top_k: int) -> dict[str, list[str]]:
    """For each query id, the ranked list of unique document ids retrieved."""
    rankings: dict[str, list[str]] = {}
    for n, (qid, qtext) in enumerate(queries.items(), start=1):
        q_emb = embedder.encode_queries([qtext])[0]
        hits = index.search(q_emb, top_k)
        rankings[qid] = dedupe_keep_order([chunk.doc_id for chunk, _ in hits])
        if n % 50 == 0:
            print(f"  ...retrieved {n}/{len(queries)} queries")
    return rankings


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Average each metric over all queries that have an answer key."""
    qids = [q for q in rankings if q in qrels]
    relevant_by_q = {
        q: {doc for doc, score in qrels[q].items() if score > 0} for q in qids
    }

    report: dict[str, float] = {}
    for k in k_values:
        report[f"Recall@{k}"] = _mean(
            [metrics.recall_at_k(rankings[q], relevant_by_q[q], k) for q in qids]
        )
        report[f"Precision@{k}"] = _mean(
            [metrics.precision_at_k(rankings[q], relevant_by_q[q], k) for q in qids]
        )
        report[f"nDCG@{k}"] = _mean(
            [metrics.ndcg_at_k(rankings[q], relevant_by_q[q], k) for q in qids]
        )
    report["MRR"] = _mean(
        [metrics.reciprocal_rank(rankings[q], relevant_by_q[q]) for q in qids]
    )
    report["#queries"] = float(len(qids))
    return report


def save_rankings(rankings: dict[str, list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rankings, f)


def load_rankings(path: Path) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
