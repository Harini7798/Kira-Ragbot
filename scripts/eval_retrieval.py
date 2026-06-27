"""Evaluate dense retrieval on SciFact's test split and print a metrics table.

Run:  python scripts/eval_retrieval.py

First run embeds every test query against the saved index and caches the
rankings to data/index/dense/rankings_test.json. Later runs reuse that cache, so
after you edit eval/metrics.py you get fresh numbers instantly.

Requires the dense index (build it with scripts/build_dense_index.py) and your
implemented metrics in src/measurable_rag/eval/metrics.py.
"""
from __future__ import annotations

from measurable_rag import config
from measurable_rag.data.qrels import load_scifact_qrels, load_scifact_queries
from measurable_rag.eval import harness

K_VALUES = (1, 5, 10)
RANKINGS_CACHE = config.INDEX_DIR / "rankings_test.json"


def main() -> None:
    qrels = load_scifact_qrels(split="test")
    all_queries = load_scifact_queries()
    # Only evaluate queries that have an answer key in this split.
    queries = {qid: all_queries[qid] for qid in qrels if qid in all_queries}
    print(f"Test split: {len(queries)} queries with relevance judgments.")

    if RANKINGS_CACHE.exists():
        print(f"Using cached rankings: {RANKINGS_CACHE}")
        rankings = harness.load_rankings(RANKINGS_CACHE)
    else:
        print("No cache yet — running retrieval (this embeds every query)...")
        from measurable_rag.retrieval.dense import DenseIndex
        from measurable_rag.retrieval.embedder import Embedder

        index = DenseIndex.load(config.INDEX_DIR)
        embedder = Embedder()
        rankings = harness.build_rankings(index, embedder, queries, config.TOP_K)
        harness.save_rankings(rankings, RANKINGS_CACHE)
        print(f"Cached rankings to {RANKINGS_CACHE}")

    try:
        report = harness.evaluate(rankings, qrels, K_VALUES)
    except NotImplementedError as e:
        print(f"\n  Metrics not implemented yet: {e}")
        print("  -> Implement the functions in src/measurable_rag/eval/metrics.py")
        print("  -> Check your work with:  python -m pytest tests/test_metrics.py")
        return

    print("\n=== Dense retrieval on SciFact (test) ===")
    n_q = int(report.pop("#queries"))
    for name, value in report.items():
        print(f"  {name:<14} {value:.4f}")
    print(f"  (averaged over {n_q} queries)")


if __name__ == "__main__":
    main()
