"""Turn text into vectors.

An embedding is a fixed-length list of numbers that captures a piece of text's
*meaning*: texts that mean similar things get vectors that point in similar
directions. That's what lets us match a question to a passage that answers it
even when they share no words.

We L2-normalize every vector (scale it to length 1). Once vectors are unit
length, their dot product equals the cosine of the angle between them — so
"nearest neighbour by dot product" becomes "most similar by cosine," which is
the standard similarity for dense retrieval.

The class keeps a separate ``encode_queries`` from ``encode_passages`` even
though all-mpnet treats them identically. Some models (e.g. BGE) need a special
instruction prepended to *queries only*; isolating that here means swapping
models later is a one-line config change, not a code change.
"""
from __future__ import annotations

import numpy as np

from .. import config


class Embedder:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        # Imported lazily so the rest of the package (chunking, loaders) stays
        # importable without the heavy PyTorch dependency installed.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or config.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name, device=device)
        # Method was renamed across sentence-transformers versions; support both.
        get_dim = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        self.dim = get_dim()
        # Whatever sentence-transformers auto-selected: 'cuda' if a CUDA GPU is
        # available (and a CUDA torch build is installed), else 'cpu'.
        self.device = str(self.model.device)

    def _encode(self, texts: list[str], show_progress: bool) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=config.EMBED_BATCH_SIZE,
            normalize_embeddings=True,      # unit vectors -> dot product == cosine
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        ).astype("float32")                 # faiss wants float32

    def encode_passages(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed corpus chunks (the documents we search over)."""
        return self._encode(texts, show_progress)

    def encode_queries(self, queries: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed search queries. Applies the model's query prefix if any."""
        if config.QUERY_PREFIX:
            queries = [config.QUERY_PREFIX + q for q in queries]
        return self._encode(queries, show_progress)
