"""Build an in-memory retriever from a set of Documents.

Used by the UI's 'upload my own documents' mode: it runs the exact same steps
the offline pipeline uses for SciFact — chunk (with offsets) -> embed -> dense
FAISS index + BM25 — but over whatever the user just uploaded, held in memory
rather than saved to disk.
"""
from __future__ import annotations

from .. import config
from ..data.chunking import chunk_corpus
from ..data.models import Document
from .dense import DenseIndex
from .pipeline import HybridRetriever
from .sparse import SparseIndex


def build_retriever(
    documents: dict[str, Document],
    embedder,
    reranker=None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> tuple[HybridRetriever, int]:
    """Chunk + embed the documents and return (retriever, n_chunks)."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    chunks = chunk_corpus(documents, chunk_size, overlap)
    if not chunks:
        raise ValueError("No text could be extracted from the uploaded documents.")

    embeddings = embedder.encode_passages([c.text for c in chunks])
    dense = DenseIndex(dim=embedder.dim)
    dense.add(chunks, embeddings)
    sparse = SparseIndex(chunks)
    return HybridRetriever(dense, sparse, embedder, reranker), len(chunks)
