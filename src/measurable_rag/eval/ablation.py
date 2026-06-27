"""The ablation runner.

Produces the centerpiece of the whole project: the table proving each retrieval
upgrade (dense -> +BM25 -> +rerank) changes the numbers. It's where the
*experiment design* lives — every config sees the exact same queries and the
exact same scoring, so any difference in the numbers is attributable to the
retrieval method alone.

See tests/test_ablation.py for the behaviour this guarantees.
"""
from __future__ import annotations

from typing import Callable

from . import harness
from ..data.models import Chunk

# A config is a name mapped to a function: given a query string, return that
# config's ranked list of chunks (best first).
Config = Callable[[str], list[Chunk]]


def run_ablation(
    configs: dict[str, Config],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    k_values: tuple[int, ...] = (1, 5, 10),
    verbose: bool = False,
) -> dict[str, dict[str, float]]:
    """Run every config over all queries and score it; return {name: report}.

    For each config:
      1. For every query, call the config function to get ranked chunks.
      2. Collapse chunks -> ranked unique DOCUMENT ids (a document's rank is the
         rank of its best chunk). The answer key (qrels) is document-level, and
         this is the same mapping the dense eval used — reuse
         ``harness.dedupe_keep_order``.
      3. Score those rankings against ``qrels`` with ``harness.evaluate``.
      4. Store the resulting metrics report under the config's name.

    The returned dict has one entry per config; each value is the metrics report
    that ``harness.evaluate`` produces (Recall@k / Precision@k / nDCG@k / MRR).
    """
    reports: dict[str, dict[str, float]] = {}
    for name, retrieve in configs.items():
        if verbose:
            print(f"  [{name}] retrieving {len(queries)} queries...", flush=True)
        # Build this config's rankings: query id -> ranked unique document ids.
        rankings: dict[str, list[str]] = {}
        for n, (qid, query_text) in enumerate(queries.items(), start=1):
            chunks = retrieve(query_text)
            # Collapse chunks -> documents; a doc keeps the rank of its best chunk.
            rankings[qid] = harness.dedupe_keep_order([c.doc_id for c in chunks])
            if verbose and n % 50 == 0:
                print(f"    ...{n}/{len(queries)}", flush=True)
        # Same queries, same scoring for every config — only `retrieve` differs.
        reports[name] = harness.evaluate(rankings, qrels, k_values)
    return reports
