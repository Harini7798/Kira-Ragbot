# Measurable RAG — "Kira"
### A retrieval-augmented QA system that measures itself and knows what it doesn't know

> **One-liner:** A full-stack RAG assistant over scientific literature that doesn't just answer —
> it reports its own retrieval quality in real benchmark numbers, **verifies every citation against
> the source**, and **refuses to answer** when the corpus can't support a claim. Built from scratch
> (no LangChain/LlamaIndex) so every retrieval internal is explicit and measurable.

---

## 1. The problem it solves

Plain RAG (chunk → embed → retrieve → answer) is commoditized and, worse, *invisible on a résumé* —
everyone has a chatbot. The harder, more valuable problem is **trust**: production chatbots
hallucinate, cite confidently and wrongly, and answer even when they have no supporting evidence.

This project makes the *measurement* the deliverable:
- **Retrieval quality** in real numbers (Recall@k, nDCG, MRR) on a labeled benchmark.
- **Citation precision** — are the cited sources actually saying what the answer claims?
- **Correct abstention** — does it refuse out-of-corpus questions instead of confabulating, and at
  what cost to questions it *could* answer?

The thesis it demonstrates: **grounding ≠ truth.** An answer can be faithfully grounded in bad
sources; the system surfaces that distinction instead of hiding it.

---

## 2. Headline results (measured on BEIR SciFact, 300-query test split)

### Retrieval ablation — proves the pipeline is non-naive
| Retrieval setup | Recall@1 | Recall@5 | Recall@10 | nDCG@10 | MRR |
|---|:--:|:--:|:--:|:--:|:--:|
| Dense embeddings (bge-small) | 0.576 | 0.770 | 0.831 | 0.713 | 0.687 |
| + BM25 hybrid (RRF fusion) | 0.570 | **0.786** | **0.851** | 0.717 | 0.684 |
| + cross-encoder reranking | *(reranker built & verified; full sweep is a one-command run)* | | | | |

The dense baseline's **nDCG@10 = 0.713 matches the published bge-small-en-v1.5 SciFact result (~0.71)** —
an external sanity check that the metrics and pipeline are implemented correctly. Hybrid fusion lifts
deeper recall (0.831 → 0.851) by catching exact-term matches the embedder smooths over.

### Citation verification — catches confident hallucination
A **separate** model judges each citation and must quote the supporting sentence (then verified to
exist in the source). On a sample query the generator produced a fluent answer whose citations were
only **~43% precise** — e.g. it cited a paper on *mulberry-leaf hepatic effects* to support a claim
about *cardiovascular disease*. The system flags this instead of presenting it as fact.

### Abstention — the standout artifact
Evaluated on 15 answerable (SciFact) + 15 plausibly-on-topic **unanswerable** questions, sweeping a
grounding threshold:
| Grounding threshold | False-answer rate ↓ | Over-refusal rate ↓ |
|:--:|:--:|:--:|
| 0.00 (permissive) | **0.000** | 0.267 |
| 0.50 | **0.000** | 0.467 |
| 1.00 (strict) | **0.000** | 0.667 |

**0% false answers across the entire operating range** — it never confabulated on an out-of-corpus
question. The key insight (in the writeup): **over-refusal is lower-bounded by retrieval recall** —
with Recall@5 = 0.77, ~23% of answerable questions simply don't have their evidence retrieved, so a
*faithful* system must decline them. Abstention quality is gated upstream by retrieval quality. The
trade-off curve is saved as a PNG/CSV.

---

## 3. What it does (the product)

A modern chat app, **"Kira"**, with three answer sources the user toggles:
- **📄 Your documents** — upload PDF / DOCX / XLSX / CSV / Markdown / code; retrieved with cited,
  verified answers and the supporting span highlighted in the source.
- **🌐 Web search** — live search (Tavily, DuckDuckGo fallback), reads the top pages, answers with
  citations to the URLs.
- **🖼️ Vision** — attach an image, ask about it (Groq multimodal model).
- **Both off → plain LLM** chat, clearly badged **⚠️ Ungrounded** so it never looks as trustworthy
  as a verified answer.

Every grounded answer shows a status badge (**✅ Grounded & verified / 🚫 Abstained / ⚠️ Ungrounded**),
per-citation ✅/❌ checks, and live citation-precision / faithfulness / grounding metrics.

Product features: email/password accounts, server-side multi-user chat history, conversation threads
(rename / search / delete / export to Markdown), staged drag-and-drop attachments, markdown rendering
with copy-able code blocks, light/dark theme, stop-generation, mobile-responsive layout, per-IP rate
limiting.

---

## 4. Architecture

```
                          React + TypeScript (Vite) front end
                                       │  REST + JWT
                                       ▼
                              FastAPI back end  ──────────────┐
   accounts (bcrypt + JWT) · SQLite (users/threads/messages)  │
   rate limiting · per-user document collections (on disk)    │
                                       │                       │
                                       ▼                       │
   query ─▶ dense (FAISS) ┐                                    │
           BM25 (rank_bm25)┼─▶ RRF fusion ─▶ cross-encoder rerank ─▶ top-k chunks
                           ┘                                    │
                                                                ▼
                          generate (Groq LLM) with [n] citations
                                                                │
                       verify each citation (independent model + span check)
                                                                │
                          grounded enough? ── yes ─▶ cited answer
                                           ── no  ─▶ abstain
```

Three layers, cleanly separated:
1. **`measurable_rag/`** — the framework-free RAG core (chunking, retrieval, fusion, rerank, metrics,
   generation, verification, abstention). Independently testable; no web/UI dependencies.
2. **`api/`** — FastAPI service exposing it: auth, threads, ask, upload; SQLite persistence.
3. **`frontend/`** — React/TS single-page chat app.

---

## 5. How it works (pipeline)

- **Chunking** — sentence-aware, overlapping chunks. Every chunk stores `(doc_id, start, end)` with
  the invariant `source[start:end] == chunk.text`, which is what makes **span-level citations** and
  source highlighting possible. Overlap prevents a fact being split across two chunks.
- **Dense retrieval** — `bge-small-en-v1.5` embeddings in a FAISS flat (exact) index; cosine via
  normalized vectors; asymmetric query-instruction prefix.
- **Sparse retrieval** — BM25 over the same chunks, for exact-term matches dense misses.
- **Fusion** — Reciprocal Rank Fusion combines the two rankings by rank position (avoids the
  incomparable-score problem of mixing cosine and BM25 scores).
- **Reranking** — a cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores the top candidates seeing
  query + passage together.
- **Generation** — a Groq-hosted LLM answers using *only* the numbered sources and cites each claim.
- **Verification** — a *different* model judges each citation and must return a verbatim supporting
  sentence, which is confirmed present in the cited chunk (else the citation is rejected). Yields
  citation precision + faithfulness.
- **Abstention** — refuse unless enough claims are grounded; the threshold is swept to trace the
  trade-off curve.

All retrieval metrics (Recall@k, nDCG@k, MRR, Precision@k) and the abstention/verification logic were
implemented **from first principles** and validated with worked-example unit tests.

---

## 6. Engineering highlights

- **No RAG framework.** Chunking, RRF fusion, the metrics, citation verification and abstention are
  all hand-written — the point was to demonstrate understanding of the internals, not glue together
  a black box.
- **Evaluator independence.** Citations are judged by a *different* model than the generator, and the
  judge must produce a verbatim quote that is checked against the source — guarding against both
  self-grading bias and a judge that hallucinates its justification.
- **Correctness validated externally.** Metric implementations were confirmed by reproducing the
  published benchmark number for the embedding model.
- **Multi-user, persistent, framework-free auth.** Email/password (bcrypt) + JWT, SQLite via
  SQLAlchemy, per-user document indexes persisted to disk so RAG corpora survive restarts.
- **Operational hardening.** Per-IP rate limiting, clean error mapping (e.g. invalid LLM key → a
  precise 401 instead of a 500), graceful degradation (web search falls back from Tavily to
  DuckDuckGo; missing docs fall back to a plain answer instead of erroring).

---

## 7. Notable problems solved (good interview stories)

- **The overlap bug a test caught.** The chunker silently produced *zero* overlap whenever sentences
  were longer than the overlap budget — defeating its own purpose. A unit test flagged it; fixed to
  always re-include at least one whole trailing sentence.
- **"Grounded ≠ true."** A web query for the latest AI model returned a confident but wrong answer
  grounded in low-quality SEO pages. Diagnosed it as a *source-quality* problem (the verifier had
  correctly flagged half the citations), then fixed it with per-domain source diversity + a
  RAG-grade search backend (Tavily) — turning a wrong answer into a correct, well-sourced one.
- **Recall bounds refusal.** Connected M2 and M5 with data: the floor on over-refusal is set by
  retrieval recall, so the real lever for fewer refusals is better retrieval, not a looser threshold.
- **UX papercuts found in a self-review.** e.g. attachments processed instantly instead of staging
  for review; links in answers navigating away from the chat; staged files leaking across
  conversations — all caught and fixed in a dedicated audit pass.

---

## 8. Tech stack

**Core (Python 3.10):** sentence-transformers (bge-small embeddings; MS-MARCO cross-encoder), FAISS,
rank-bm25, NumPy, Groq API (Llama 3.x generation + 8B verifier + Llama-4 vision), Tavily search,
PyTorch (CPU/GPU). **Backend:** FastAPI, SQLAlchemy + SQLite, PyJWT, bcrypt, slowapi, Uvicorn.
**Frontend:** React 18, TypeScript, Vite, react-markdown. **Quality:** pytest (33 tests),
src-layout package, pyproject. **Deliberately avoided:** LangChain / LlamaIndex.

---

## 9. Testing & quality

33 unit tests covering the metric math (with hand-computed expected values), the chunker's offset
invariant and overlap, the ablation runner, citation parsing/precision, and the abstention threshold
sweep. The retrieval pipeline's correctness is additionally validated against the published SciFact
benchmark figure.

---

## 10. Limitations & future work (honest)

- **Citation judge is an LLM**, not a fine-tuned NLI model — a different model reduces self-bias, but
  a dedicated NLI checker would be more rigorous (the code is structured for a drop-in swap).
- **Unanswerable eval slice is hand-curated** — a stronger construction holds each answerable claim's
  gold documents out of the index and re-asks it.
- **Deployment** to a public URL is the remaining step (Hugging Face Spaces is the planned target,
  given the ~2-3 GB ML footprint).
- **Streaming responses** (token-by-token) is the one mainstream chat feature not yet implemented
  (needs an SSE endpoint + reworking the verify-after-answer flow).

---

## 11. Résumé bullet points

Pick the 2–4 that fit the role; the first three are the strongest.

- Built a **retrieval-augmented QA system** over 5,000+ scientific documents that self-measures
  retrieval quality, achieving **Recall@5 = 0.79 and nDCG@10 = 0.71** on the BEIR SciFact benchmark
  (matching published results for the embedding model) via **hybrid dense + BM25 retrieval with
  reciprocal-rank fusion and cross-encoder reranking — implemented from scratch without RAG
  frameworks.**
- Designed an **independent citation-verification pass** (a separate LLM judge that checks every
  generated citation against its exact source span), **exposing that raw model citations were only
  ~43% precise** — converting silent hallucination into a reported, auditable metric.
- Engineered **grounded abstention** with a tunable threshold: **0% false-answer rate** on
  out-of-corpus questions while quantifying the false-answer vs. over-refusal trade-off, and showed
  **over-refusal is lower-bounded by retrieval recall** — linking two pipeline stages with data.
- Implemented retrieval metrics (**Recall@k, nDCG@k, MRR, Precision@k**) and a sentence-aware chunker
  with exact character-offset preservation from first principles; **33 unit tests**, correctness
  validated against published benchmark numbers.
- Shipped a **full-stack chat application** (React + TypeScript front end, FastAPI back end) with
  **email/password auth (bcrypt + JWT), SQLite multi-user persistence**, document RAG
  (PDF/DOCX/XLSX), **live web search**, image/vision Q&A, and per-IP rate limiting.
- Hardened for production: graceful fallbacks (search-provider failover, no-evidence → plain answer),
  precise error mapping, and source-diversity fixes that turned wrong web answers into correct,
  well-cited ones.

**One-line project description (résumé header):**
> *Measurable RAG (“Kira”) — a full-stack retrieval-QA assistant with verified span-level citations
> and grounded abstention; hybrid retrieval + reranking reaching Recall@5 0.79 on BEIR SciFact, 0%
> false-answer rate on out-of-corpus questions.*

**Skills demonstrated:** Information retrieval & evaluation · RAG architecture · LLM application
engineering · embeddings / vector search (FAISS) · BM25 · cross-encoder reranking · prompt design ·
LLM-as-judge evaluation · full-stack (React/TS + FastAPI) · auth & multi-user data modeling ·
testing & benchmarking · Python.

---

## 12. Interview talking points

- *Why it's not just a chatbot:* it reports measured retrieval quality, verifies its own citations,
  and abstains — and I can defend each number.
- *Grounding vs. truth:* faithfulness to sources ≠ correctness; source quality is a separate, explicit
  knob, which is why I added source diversity + a better search backend.
- *Why hybrid + rerank:* the ablation table shows each component's contribution on the same queries
  and scoring — attributable, not hand-wavy.
- *Why abstention is hard to fake:* the unanswerable questions are plausibly on-topic, so refusing
  them is genuinely difficult; and the over-refusal floor is set by retrieval recall.
