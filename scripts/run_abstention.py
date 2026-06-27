"""Run the abstention experiment and produce the false-answer vs over-refusal curve.

Run:  python scripts/run_abstention.py

For every eval question (answerable + unanswerable): retrieve, generate, verify
the citations, and record the answer's grounding score. Then sweep the grounding
threshold to trace the trade-off curve, print it, save the records + sweep, and
(if matplotlib is installed) save a PNG plot. Requires GROQ_API_KEY.

This is the project's headline artifact — it shows the system refusing
out-of-corpus questions without over-refusing answerable ones, and exposes the
tunable knob that trades the two off.
"""
from __future__ import annotations

import csv
import json

from measurable_rag import config
from measurable_rag.console import use_utf8
from measurable_rag.data.eval_set import build_eval_set
from measurable_rag.eval.abstention import AbstentionRecord, best_threshold, sweep
from measurable_rag.generation.abstain import grounding_score
from measurable_rag.generation.generator import INSUFFICIENT, Generator
from measurable_rag.generation.verify import Verifier
from measurable_rag.retrieval.dense import DenseIndex
from measurable_rag.retrieval.embedder import Embedder
from measurable_rag.retrieval.pipeline import HybridRetriever
from measurable_rag.retrieval.sparse import SparseIndex

RESULTS_DIR = config.PROJECT_ROOT / "results"
THRESHOLDS = [i / 20 for i in range(21)]  # 0.00, 0.05, ... 1.00


def build_records(items, retriever, generator, verifier) -> list[AbstentionRecord]:
    records = []
    for n, item in enumerate(items, start=1):
        chunks = retriever.hybrid(item["question"], config.GENERATION_TOP_K)
        answer = generator.answer(item["question"], chunks)
        insufficient = answer.strip().startswith("INSUFFICIENT")
        verdicts = [] if insufficient else verifier.verify(answer, chunks)
        records.append(
            AbstentionRecord(
                qid=item["id"],
                label=item["label"],
                grounding=grounding_score(verdicts),
                insufficient=insufficient,
            )
        )
        print(f"  [{n}/{len(items)}] {item['label']:<12} grounding="
              f"{records[-1].grounding:.2f} insufficient={insufficient}", flush=True)
    return records


def save_outputs(records, rows) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "abstention_records.json", "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in records], f, indent=2)
    with open(RESULTS_DIR / "abstention_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["threshold", "false_answer_rate", "over_refusal_rate"])
        w.writeheader()
        w.writerows(rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 5))
        plt.plot([r["threshold"] for r in rows], [r["false_answer_rate"] for r in rows],
                 marker="o", label="false-answer rate (unanswerable answered)")
        plt.plot([r["threshold"] for r in rows], [r["over_refusal_rate"] for r in rows],
                 marker="s", label="over-refusal rate (answerable refused)")
        plt.xlabel("grounding threshold")
        plt.ylabel("error rate")
        plt.title("Abstention trade-off")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(RESULTS_DIR / "abstention_tradeoff.png", dpi=130, bbox_inches="tight")
        print(f"Saved plot -> {RESULTS_DIR / 'abstention_tradeoff.png'}")
    except ImportError:
        print("(install matplotlib to also save a PNG plot)")


def main() -> None:
    use_utf8()
    items = build_eval_set()
    n_ans = sum(1 for i in items if i["label"] == "answerable")
    n_una = len(items) - n_ans
    print(f"Eval set: {n_ans} answerable + {n_una} unanswerable = {len(items)} questions.")
    print("Generating + verifying (this makes several LLM calls per question)...")

    index = DenseIndex.load(config.INDEX_DIR)
    retriever = HybridRetriever(index, SparseIndex(index.chunks), Embedder())
    records = build_records(items, retriever, Generator(), Verifier())

    rows = sweep(records, THRESHOLDS)
    save_outputs(records, rows)

    print("\n=== Abstention trade-off (SciFact answerable + curated unanswerable) ===")
    print(f"{'threshold':>9}  {'false-answer':>12}  {'over-refusal':>12}")
    for r in rows:
        print(f"{r['threshold']:>9.2f}  {r['false_answer_rate']:>12.3f}  {r['over_refusal_rate']:>12.3f}")

    best = best_threshold(rows)
    print(f"\nBalanced operating point: threshold={best['threshold']:.2f}  "
          f"false-answer={best['false_answer_rate']:.3f}  "
          f"over-refusal={best['over_refusal_rate']:.3f}")
    print(f"Saved records + sweep CSV -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
