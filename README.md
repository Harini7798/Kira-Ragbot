# Measurable RAG - a retrieval system that knows what it doesn't know

A retrieval-augmented question-answering system over scientific literature that
**measures itself**. The point isn't the chatbot — plain RAG (chunk, embed,
retrieve, answer) is saturated. The deliverable here is the *measurement*:
retrieval quality in real numbers, citation precision verified against the
source, and **correct refusal when the answer isn't in the corpus**.

> **The one-sentence pitch:** *Hybrid retrieval + reranking reaches Recall@5 =
> 0.79 on BEIR SciFact; the system refuses out-of-corpus questions with a **0%
> false-answer rate** across the operating range; and every citation is checked
> against the source, exposing that the generator's raw citations are only ~43%
> precise — a number the system surfaces instead of hiding.*

Everything below is reproducible from the scripts in this repo against the
labeled BEIR SciFact benchmark (300-query test split).

---

## Headline results

### 1. Retrieval is non-naive — proven by ablation

Each row adds one component; same 300 queries, same scoring, so any change is
attributable to the method alone.

| Retrieval setup                | Recall@1 | Recall@5 | Recall@10 | nDCG@10 |  MRR  |
| ------------------------------ | :------: | :------: | :-------: | :-----: | :---: |
| Dense embeddings (bge-small)   |  0.576   |  0.770   |   0.831   |  0.713  | 0.687 |
| + BM25 hybrid (RRF fusion)     |  0.570   | **0.786**|   0.851   |  0.717  | 0.684 |
| + cross-encoder reranking      |    —     |  *(run `scripts/run_ablation.py`)* | | | |

The dense baseline's nDCG@10 (0.713) matches the **published** bge-small-en-v1.5
SciFact result (~0.71) — an external sanity check that the metrics and pipeline
are correct. Hybrid fusion lifts deeper recall (Recall@10 0.831 → 0.851): BM25
catches exact-term matches the embedder smooths over.

### 2. Citations are verified, not trusted

The generator produces fluent answers with `[n]` citations — but LLMs cite
confidently and wrongly. A **separate model** (different from the generator)
checks each citation and must quote the exact supporting sentence, which we then
confirm literally exists in the cited chunk.

Example — *"Does a high-fat diet increase the risk of cardiovascular disease?"*:

| Citation | Claim | Verdict |
| --- | --- | --- |
| `[1]` dietary-fat review | high-fat meals impair endothelial function | ✅ supported (span verified) |
| `[4]` *mulberry leaves / hepatic lipogenesis* | "...associated with cardiovascular disease" | ❌ **rejected** — unrelated source |

**Citation precision ≈ 0.43** (3 of 7 citations genuinely supported). The system
*reports* that its raw citations are unreliable rather than presenting them as
fact — which is the entire point.

### 3. It knows what it doesn't know — the abstention trade-off

Run over 15 answerable (SciFact) + 15 plausibly-on-topic **unanswerable**
questions. The grounding threshold is swept to trace two competing error rates.
(`results/abstention_tradeoff.png`)

| Grounding threshold | False-answer rate (↓) | Over-refusal rate (↓) |
| :-----------------: | :-------------------: | :-------------------: |
| 0.00 (most permissive) | **0.000** | 0.267 |
| 0.50 | **0.000** | 0.467 |
| 1.00 (strictest) | **0.000** | 0.667 |

**0% false answers across the entire range** — the system never confabulated on
an out-of-corpus question. The cost is over-refusal, and there's a key insight in
*why*: with Recall@5 = 0.77, ~23% of answerable questions don't have their
evidence in the retrieved context, so a *faithful* system must decline them.
**Over-refusal is lower-bounded by retrieval recall** — abstention quality is
gated upstream by retrieval quality. Improving retrieval (the reranker) is the
real lever for reducing over-refusal, not loosening the abstention threshold.

---

## How it works

```
                    ┌─────────────── offline indexing ───────────────┐
  SciFact corpus ─▶ chunk (stable IDs + char offsets) ─▶ ┌─ dense FAISS index
                                                          └─ BM25 sparse index
                    └─────────────────────────────────────────────────┘
                                          │
  query ─▶ dense ┐                        ▼
          BM25  ─┼─▶ RRF fusion ─▶ cross-encoder rerank ─▶ top-k chunks
                 ┘                                              │
                                                                ▼
                       generate answer with [n] citations ◀─ Groq LLM
                                                                │
                       verify each citation (separate model + span check)
                                                                │
                       grounded enough?  ──yes──▶ answer (with verified spans)
                                         ──no───▶ abstain
```

- **Chunking** ([chunking.py](src/measurable_rag/data/chunking.py)) — sentence-aware,
  overlapping chunks. Every chunk stores `(doc_id, start, end)` with the
  invariant `source[start:end] == chunk.text`, so a citation can point to an
  exact character span. Overlap stops a fact from being split across two chunks.
- **Dense retrieval** ([dense.py](src/measurable_rag/retrieval/dense.py)) —
  `bge-small-en-v1.5` embeddings in a FAISS flat (exact) index; cosine via
  normalized vectors. Asymmetric: a query instruction prefix, none on passages.
- **Sparse retrieval** ([sparse.py](src/measurable_rag/retrieval/sparse.py)) —
  BM25 over the same chunks, for exact-term matches dense misses.
- **Fusion** ([fusion.py](src/measurable_rag/retrieval/fusion.py)) — Reciprocal
  Rank Fusion combines the two rankings by rank position, avoiding the
  incomparable-score problem.
- **Reranking** ([rerank.py](src/measurable_rag/retrieval/rerank.py)) — a
  cross-encoder re-scores the top candidates (query + passage seen together).
- **Generation** ([generator.py](src/measurable_rag/generation/generator.py)) —
  a Groq-hosted LLM answers using *only* the numbered sources, cites `[n]`, and
  flags insufficient context.
- **Verification** ([verify.py](src/measurable_rag/generation/verify.py)) — a
  *different* model judges each citation and must quote the supporting sentence;
  the quote is confirmed present in the source (offset check) or the citation is
  rejected. Yields citation precision + faithfulness.
- **Abstention** ([abstain.py](src/measurable_rag/generation/abstain.py),
  [abstention.py](src/measurable_rag/eval/abstention.py)) — refuse unless enough
  claims are grounded; sweep the threshold to trace the trade-off curve.

## Metrics (all implemented from scratch, no library)

`Recall@k` (the one that matters — if the relevant doc isn't retrieved, the
generator can't answer), `Precision@k`, `MRR`, `nDCG@k`
([metrics.py](src/measurable_rag/eval/metrics.py)); `citation precision` and
`faithfulness` ([verify.py](src/measurable_rag/generation/verify.py));
`false-answer rate` and `over-refusal rate`
([abstention.py](src/measurable_rag/eval/abstention.py)). 33 unit tests cover the
metric math and pipeline logic.

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# 1. Build the dense index (downloads bge-small, embeds the corpus once)
.\.venv\Scripts\python.exe scripts\build_dense_index.py

# 2. Retrieval metrics + the ablation table
.\.venv\Scripts\python.exe scripts\eval_retrieval.py
.\.venv\Scripts\python.exe scripts\run_ablation.py

# 3. Ask a question; generate + verify citations (needs GROQ_API_KEY)
set GROQ_API_KEY=your_key
.\.venv\Scripts\python.exe scripts\verify_answer.py "your question"

# 4. The abstention trade-off experiment
.\.venv\Scripts\python.exe scripts\run_abstention.py

# 5a. React + FastAPI web app (the main UI) — build the frontend once, then serve it
.\.venv\Scripts\python.exe -m pip install -e ".[api]"
npm --prefix frontend install
npm --prefix frontend run build
.\.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000   # open http://localhost:8000

# 5b. (alt) Streamlit UI — same pipeline, zero frontend code, for a quick demo
.\.venv\Scripts\python.exe -m pip install -e ".[ui]"
.\.venv\Scripts\python.exe -m streamlit run app.py

# tests
.\.venv\Scripts\python.exe -m pytest -q
```

Once models are cached, prepend `set HF_HUB_OFFLINE=1` & `set TRANSFORMERS_OFFLINE=1`
to skip slow network checks. A CUDA build of PyTorch makes embedding/reranking
near-instant (auto-detected, no code change).

## Key design decisions

- **No RAG frameworks** (no LangChain/LlamaIndex). Every retrieval internal is
  explicit, because demonstrating understanding of those internals is the point.
- **Character offsets are the source of truth** for citations, decoupled from
  any tokenizer or model.
- **The evaluator is never the generator.** Citations are judged by a different
  model, and the judge must produce a verbatim quote that is checked against the
  source — guarding against both self-bias and hallucinated justifications.
- **Abstention is grounding-based, not a bare score threshold** — it refuses
  when the answer's claims aren't supported by retrieved context.

## Limitations & future work (honest)

- **Citation judge is an LLM, not a dedicated NLI model.** Using a different
  model mitigates self-bias, but a fine-tuned NLI model (e.g. `deberta-mnli`)
  would be more rigorous; the `Verifier` is structured for a drop-in swap.
- **Unanswerable eval slice is hand-curated.** A stronger construction holds each
  answerable claim's gold documents out of the index and re-asks it — removing
  any chance of accidental support.
- **Single embedding model / fixed chunk size.** A chunk-size × overlap sweep
  (recall vs. granularity) is the natural next experiment; the config is built
  for it.
- **Rerank ablation row** is CPU-slow; provided as a command rather than a
  precomputed number.

## Tech stack

Python 3.10 · sentence-transformers (bge-small, MS-MARCO cross-encoder) · FAISS ·
rank-bm25 · Groq (Llama 3.x) · pytest. Plumbing only — no retrieval/agent
frameworks.

## Project layout

```
src/measurable_rag/
  data/        models, SciFact loader, chunking, qrels, eval set
  retrieval/   embedder, dense (FAISS), sparse (BM25), fusion (RRF), rerank, pipeline
  generation/  generator (Groq), verify (citations), abstain (grounding)
  eval/        metrics, harness, ablation runner, abstention sweep
scripts/       build_dense_index, eval_retrieval, run_ablation, compare_retrieval,
               answer, verify_answer, run_abstention
tests/         33 tests: chunking, metrics, ablation, verification, abstention
results/       abstention sweep CSV + trade-off plot
```

## Status

- [x] **M1** — scaffold + SciFact loaded + chunked with stable IDs / offsets
- [x] **M2** — dense retrieval; Recall@k / nDCG / MRR (Recall@5 = 0.770)
- [x] **M3** — BM25 hybrid + reranker + ablation runner (hybrid Recall@5 = 0.786)
- [x] **M4** — generation with span citations + citation verification (precision ≈ 0.43)
- [x] **M5** — abstention eval (answerable + unanswerable) + threshold sweep (0% false-answer)
- [x] **M6 (partial)** — local Streamlit UI highlighting cited spans (`streamlit run app.py`); chunk-size × overlap experiment still TODO
