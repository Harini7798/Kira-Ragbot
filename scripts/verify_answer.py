"""Ask a question, generate a cited answer, then VERIFY every citation.

Run:  python scripts/verify_answer.py "your question"

Shows the answer, each claim->source citation with a SUPPORTED/UNSUPPORTED
verdict (judged by a different model), the exact supporting span when found, and
the overall citation precision + faithfulness. Requires GROQ_API_KEY.
"""
from __future__ import annotations

import sys

from measurable_rag import config
from measurable_rag.console import use_utf8
from measurable_rag.generation.generator import Generator
from measurable_rag.generation.verify import (
    Verifier,
    citation_precision,
    faithfulness,
)
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

    answer = Generator().answer(query, chunks)
    print(f"Question: {query}\n\n--- Answer ---\n{answer}\n")

    verdicts = Verifier().verify(answer, chunks)
    print("--- Citation verification ---")
    for v in verdicts:
        mark = "OK " if v.supported else "BAD"
        print(f"  [{mark}] [{v.source_num}] (doc {v.doc_id})  claim: {v.claim[:70]!r}")
        if v.supported and v.quote:
            print(f"         supported by span {v.span}: {v.quote[:90]!r}")

    cp = citation_precision(verdicts)
    print("\n--- Metrics ---")
    print(f"  citations checked : {len(verdicts)}")
    print(f"  citation precision: {cp:.3f}" if cp is not None else "  citation precision: n/a")
    print(f"  faithfulness      : {faithfulness(verdicts):.3f}")


if __name__ == "__main__":
    main()
