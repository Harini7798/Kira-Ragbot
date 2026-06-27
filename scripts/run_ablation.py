"""Run the retrieval ablation on SciFact and print the comparison table.

Run:  python scripts/run_ablation.py

Builds the three configs (dense / +BM25 hybrid / +rerank) over the saved index
and your implemented run_ablation, then prints Recall@k / nDCG / MRR for each.

Needs: the dense index (scripts/build_dense_index.py), the reranker model
downloaded, and run_ablation implemented in src/measurable_rag/eval/ablation.py.
Reranking 50 candidates x 300 queries is the slow part on CPU (several minutes);
it's much faster if you've switched to the CUDA build of PyTorch.
"""
from __future__ import annotations

from measurable_rag import config
from measurable_rag.console import use_utf8
from measurable_rag.data.qrels import load_scifact_qrels, load_scifact_queries
from measurable_rag.eval.ablation import run_ablation
from measurable_rag.retrieval.dense import DenseIndex
from measurable_rag.retrieval.embedder import Embedder
from measurable_rag.retrieval.pipeline import HybridRetriever
from measurable_rag.retrieval.rerank import Reranker
from measurable_rag.retrieval.sparse import SparseIndex

K_VALUES = (1, 5, 10)
COLUMNS = ["Recall@1", "Recall@5", "Recall@10", "nDCG@10", "MRR"]


def print_table(reports: dict[str, dict[str, float]]) -> None:
    name_w = max(len(n) for n in reports)
    header = "config".ljust(name_w) + "  " + "  ".join(c.rjust(9) for c in COLUMNS)
    print("\n=== Retrieval ablation on SciFact (test) ===")
    print(header)
    print("-" * len(header))
    for name, rep in reports.items():
        row = name.ljust(name_w) + "  " + "  ".join(f"{rep[c]:9.4f}" for c in COLUMNS)
        print(row)


def main() -> None:
    use_utf8()
    qrels = load_scifact_qrels(split="test")
    all_queries = load_scifact_queries()
    queries = {qid: all_queries[qid] for qid in qrels if qid in all_queries}
    print(f"Test split: {len(queries)} queries.")

    print("Loading index, embedder, BM25, reranker ...")
    index = DenseIndex.load(config.INDEX_DIR)
    embedder = Embedder()
    sparse = SparseIndex(index.chunks)
    reranker = Reranker()
    retriever = HybridRetriever(index, sparse, embedder, reranker)

    configs = {
        "dense": lambda q: retriever.dense(q),
        "hybrid (dense+BM25)": lambda q: retriever.hybrid(q),
        "hybrid + rerank": lambda q: retriever.hybrid_reranked(q),
    }

    print("Running ablation (the rerank config is the slow one)...")
    try:
        reports = run_ablation(configs, queries, qrels, K_VALUES, verbose=True)
    except NotImplementedError as e:
        print(f"\n  run_ablation not implemented yet: {e}")
        print("  -> Implement it in src/measurable_rag/eval/ablation.py")
        print("  -> Check with:  python -m pytest tests/test_ablation.py -q")
        return

    print_table(reports)


if __name__ == "__main__":
    main()
