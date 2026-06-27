"""Embed the whole SciFact corpus and build + save a dense FAISS index.

Run:  python scripts/build_dense_index.py

Embeds once and writes to data/index/dense/. Re-run only if the corpus, chunk
settings, or embedding model change. Ends with a sanity query so you can see
retrieval actually returning relevant chunks.
"""
from __future__ import annotations

import time

from measurable_rag import config
from measurable_rag.console import use_utf8
from measurable_rag.data.chunking import chunk_corpus
from measurable_rag.data.loaders import load_scifact_corpus
from measurable_rag.retrieval.dense import DenseIndex
from measurable_rag.retrieval.embedder import Embedder


def main() -> None:
    use_utf8()
    docs = load_scifact_corpus()
    chunks = chunk_corpus(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"Loaded {len(docs):,} docs -> {len(chunks):,} chunks.")

    print(f"Loading embedding model: {config.EMBEDDING_MODEL} ...")
    embedder = Embedder()
    print(f"Embedding dimension: {embedder.dim}  |  device: {embedder.device}")

    print(f"Embedding {len(chunks):,} chunks (this is the slow part) ...")
    t0 = time.perf_counter()
    embeddings = embedder.encode_passages([c.text for c in chunks], show_progress=True)
    print(f"Embedded in {time.perf_counter() - t0:.0f}s "
          f"-> matrix {embeddings.shape}")

    index = DenseIndex(dim=embedder.dim)
    index.add(chunks, embeddings)
    index.save(config.INDEX_DIR)
    print(f"Saved index ({index.index.ntotal:,} vectors) to {config.INDEX_DIR}")

    # --- sanity check: does a query return sensible chunks? -----------------
    sanity_query = "Does a high-fat diet increase the risk of cardiovascular disease?"
    print(f"\nSanity query: {sanity_query!r}")
    q_emb = embedder.encode_queries([sanity_query])[0]
    for rank, (chunk, score) in enumerate(index.search(q_emb, k=3), start=1):
        preview = chunk.text[:160].replace("\n", " ")
        print(f"  #{rank}  score={score:.3f}  doc={chunk.doc_id}  {preview!r}")


if __name__ == "__main__":
    main()
