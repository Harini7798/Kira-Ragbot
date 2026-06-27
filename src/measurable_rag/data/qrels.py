"""Load SciFact's queries and relevance judgments (qrels).

Two pieces, both from the BEIR download:

* ``queries.jsonl`` — every query (a scientific claim) with an id.
* ``qrels/<split>.tsv`` — the answer key: rows of (query-id, corpus-id, score)
  saying which corpus documents are relevant to which query. SciFact scores are
  binary (1 = relevant).

The qrels are keyed by query id and corpus *document* id. Our retriever returns
*chunks*, so the eval harness will map chunks back to their document ids before
comparing against this answer key.
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import config


def load_scifact_queries(raw_dir: Path | None = None) -> dict[str, str]:
    """All queries as ``{query_id: text}``."""
    raw_dir = raw_dir or config.RAW_DIR
    path = raw_dir / "scifact" / "queries.jsonl"
    queries: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            queries[str(obj["_id"])] = obj["text"]
    return queries


def load_scifact_qrels(
    split: str = "test", raw_dir: Path | None = None
) -> dict[str, dict[str, int]]:
    """Answer key as ``{query_id: {doc_id: score}}`` for the given split.

    Only queries that appear here should be evaluated — the other split's
    queries have no judgments, so we can't score them.
    """
    raw_dir = raw_dir or config.RAW_DIR
    path = raw_dir / "scifact" / "qrels" / f"{split}.tsv"
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            qid, doc_id, score = parts
            if i == 0 and not score.lstrip("-").isdigit():
                continue  # skip the "query-id corpus-id score" header row
            qrels.setdefault(str(qid), {})[str(doc_id)] = int(score)
    return qrels
