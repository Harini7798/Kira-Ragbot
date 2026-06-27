# syntax=docker/dockerfile:1
# Single-image deploy: build the React frontend, then run FastAPI which serves it.

# ---- Stage 1: build the React frontend -> frontend/dist ----
FROM node:20-slim AS frontend
WORKDIR /app/frontend
# .dockerignore excludes node_modules/dist, so this copies just the source.
COPY frontend/ ./
RUN npm install && npm run build

# ---- Stage 2: Python backend ----
FROM python:3.10-slim
WORKDIR /app

# libgomp1 is needed by faiss / torch (OpenMP); build-essential covers any
# source-only wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install the package + API extras (pulls torch, sentence-transformers, faiss, …)
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[api]"

# Bake the embedding + reranker models INTO the image so runtime is fast and
# fully offline (no Hugging Face calls in production).
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# App code + the prebuilt frontend
COPY api/ ./api/
COPY --from=frontend /app/frontend/dist ./frontend/dist

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1
EXPOSE 8000
# Render injects $PORT; default to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
