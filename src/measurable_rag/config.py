"""Central configuration: tunable knobs in one place.

Keeping these here (rather than scattered as magic numbers) means your later
experiments — the chunk-size/overlap sweep, swapping embedding models — change
one file, and every run records which settings produced which numbers.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths ------------------------------------------------------------------
# config.py lives at <root>/src/measurable_rag/config.py, so the repo root is
# three parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from a project-root .env into the environment.

    Keeps the Groq API key in a gitignored .env file rather than in the UI or in
    committed source. Existing env vars win (setdefault), so you can still
    override by exporting GROQ_API_KEY in your shell.
    """
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, v = s.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

# --- Chunking ---------------------------------------------------------------
# Measured in CHARACTERS for now. Character offsets are what we cite either way
# (they're the source of truth for a span), and char-based sizing needs no
# tokenizer dependency. Once we pick an embedding model in M2 we can switch the
# size *unit* to tokens to match its context budget, without touching offsets.
CHUNK_SIZE = 800       # target max chunk length in characters
CHUNK_OVERLAP = 150    # how much adjacent chunks overlap, in characters

# --- Datasets ---------------------------------------------------------------
# Canonical BEIR mirror. SciFact is small (~5k abstracts, a few MB).
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

# --- Dense retrieval (M2) ---------------------------------------------------
# bge-small: small + fast on CPU, strong on BEIR. It's an ASYMMETRIC model — it
# was trained expecting a short instruction prepended to QUERIES only (never to
# passages), so queries and passages are encoded differently. The embedder
# applies QUERY_PREFIX in encode_queries() and nothing in encode_passages().
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBED_BATCH_SIZE = 64
INDEX_DIR = DATA_DIR / "index" / "dense"
# Retrieve a wide top-K cheaply; later milestones rerank this down to a few.
TOP_K = 50

# --- Hybrid retrieval + reranking (M3) --------------------------------------
# Reciprocal Rank Fusion constant. Larger = flatter weighting across ranks; 60
# is the value from the original RRF paper and the common default.
RRF_K = 60
# Small, fast cross-encoder trained on MS MARCO passage ranking. Reranks the
# fused candidate pool below down to the final top-k.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_POOL = TOP_K  # how many fused candidates to feed the reranker

# --- Generation (M4) --------------------------------------------------------
# Groq serves open models over an OpenAI-compatible API. The LLM is a swappable
# component — the retrieval + measurement around it is the actual project.
# The API key is read from the GROQ_API_KEY environment variable (never stored
# in code), so it stays out of the repo.
GROQ_MODEL = "llama-3.3-70b-versatile"
GENERATION_TOP_K = 5  # how many retrieved chunks to hand the generator as sources
# Multimodal model for image questions (Groq). Swap if Groq rotates model ids.
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# --- Verification + abstention (M4 / M5) ------------------------------------
# A DIFFERENT model judges the citations than the one that wrote the answer, so
# we don't let a model grade its own work (the evaluator-bias pitfall). A
# dedicated NLI model would be even more independent — noted as future work.
VERIFIER_MODEL = "llama-3.1-8b-instant"
# Abstain unless at least this fraction of the answer's claims are grounded in
# their cited sources. This threshold is what the M5 experiment sweeps.
ABSTAIN_THRESHOLD = 0.5
