"""A dense (vector) index over chunks, backed by FAISS.

FAISS stores many vectors and answers "which stored vectors are nearest to this
query vector?" quickly. At our scale (~15k chunks) we use ``IndexFlatIP`` — a
*flat* (exhaustive) index that compares the query against every vector. It's
exact (no approximation error) and instant at this size. Approximate indexes
(IVF, HNSW) only earn their complexity at millions of vectors.

"IP" = inner product. Because the embedder hands us unit-length vectors, inner
product == cosine similarity, and FAISS returns results in descending
similarity — i.e. most-similar first.

Alongside the vectors we keep a parallel list of the ``Chunk`` objects: vector
``i`` corresponds to ``chunks[i]``, so a search result's row number maps straight
back to the chunk (and thus its document ID and character offsets).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data.models import Chunk


class DenseIndex:
    def __init__(self, dim: int):
        import faiss

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings must line up 1:1")
        self.index.add(embeddings.astype("float32"))
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        """Return the top-k (chunk, similarity) pairs, most similar first."""
        q = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        scores, idxs = self.index.search(q, k)
        results = []
        for score, i in zip(scores[0], idxs[0]):
            if i == -1:  # faiss pads with -1 if fewer than k vectors exist
                continue
            results.append((self.chunks[i], float(score)))
        return results

    # --- persistence --------------------------------------------------------
    # Vectors -> a faiss binary file; chunk metadata -> jsonl (human-readable,
    # and the line order matches the vector order so they stay aligned).
    def save(self, index_dir: Path) -> None:
        import faiss

        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "dense.faiss"))
        with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps({
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "start": c.start,
                    "end": c.end,
                    "text": c.text,
                }) + "\n")

    @classmethod
    def load(cls, index_dir: Path) -> "DenseIndex":
        import faiss

        index = faiss.read_index(str(index_dir / "dense.faiss"))
        obj = cls.__new__(cls)  # bypass __init__ (it would build an empty index)
        obj.index = index
        obj.dim = index.d
        obj.chunks = []
        with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                obj.chunks.append(Chunk(**d))
        if len(obj.chunks) != index.ntotal:
            raise ValueError("index/chunks size mismatch — rebuild the index")
        return obj
