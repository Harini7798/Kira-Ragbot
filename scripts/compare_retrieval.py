"""See dense vs. BM25 vs. hybrid side by side for one query.

Run:  python scripts/compare_retrieval.py [your query words...]

A debugging/intuition tool — it shows WHY fusion helps: dense and BM25 often
surface different chunks, and the hybrid blends them.
"""
from __future__ import annotations

import sys

from measurable_rag import config
from measurable_rag.console import use_utf8
from measurable_rag.retrieval.dense import DenseIndex
from measurable_rag.retrieval.embedder import Embedder
from measurable_rag.retrieval.fusion import reciprocal_rank_fusion
from measurable_rag.retrieval.sparse import SparseIndex

DEFAULT_QUERY = "Does a high-fat diet increase the risk of cardiovascular disease?"


def show(title: str, chunks: list, n: int = 3) -> None:
    print(f"\n{title}")
    for rank, c in enumerate(chunks[:n], start=1):
        print(f"  #{rank}  doc={c.doc_id:<10}  {c.text[:95]!r}")


def main() -> None:
    use_utf8()
    query = " ".join(sys.argv[1:]) or DEFAULT_QUERY
    print(f"Query: {query!r}")

    index = DenseIndex.load(config.INDEX_DIR)
    sparse = SparseIndex(index.chunks)  # same chunks -> fair comparison
    embedder = Embedder()

    dense = [c for c, _ in index.search(embedder.encode_queries([query])[0], config.TOP_K)]
    bm25 = [c for c, _ in sparse.search(query, config.TOP_K)]
    hybrid = reciprocal_rank_fusion([dense, bm25])

    show("DENSE  (matches by meaning):", dense)
    show("BM25   (matches by keywords):", bm25)
    show("HYBRID (reciprocal rank fusion):", hybrid)


if __name__ == "__main__":
    main()
