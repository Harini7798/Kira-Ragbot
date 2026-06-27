"""Ask the RAG system a question: retrieve, then generate a cited answer.

Run:  python scripts/answer.py "your question here"

Requires GROQ_API_KEY in the environment. Uses hybrid retrieval (dense + BM25)
to pick the sources, then Groq to write the answer.
"""
from __future__ import annotations

import sys

from measurable_rag import config
from measurable_rag.console import use_utf8
from measurable_rag.generation.generator import Generator
from measurable_rag.retrieval.dense import DenseIndex
from measurable_rag.retrieval.embedder import Embedder
from measurable_rag.retrieval.pipeline import HybridRetriever
from measurable_rag.retrieval.sparse import SparseIndex

DEFAULT_Q = "Does a high-fat diet increase the risk of cardiovascular disease?"


def main() -> None:
    use_utf8()
    query = " ".join(sys.argv[1:]) or DEFAULT_Q

    index = DenseIndex.load(config.INDEX_DIR)
    retriever = HybridRetriever(index, SparseIndex(index.chunks), Embedder())
    chunks = retriever.hybrid(query, config.GENERATION_TOP_K)

    print(f"Question: {query}\n")
    print(f"Retrieved {len(chunks)} sources:")
    for i, c in enumerate(chunks, start=1):
        print(f"  [{i}] doc {c.doc_id}: {c.text[:90]!r}")

    generator = Generator()
    print("\n--- Answer ---")
    print(generator.answer(query, chunks))


if __name__ == "__main__":
    main()
