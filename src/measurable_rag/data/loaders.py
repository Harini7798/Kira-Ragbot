"""Corpus loaders. SciFact first; PDF loader will join here later.

SciFact ships in BEIR's standard layout: a zip containing
``scifact/corpus.jsonl`` (the documents), ``queries.jsonl``, and ``qrels/``
(the relevance judgments we'll need in M2). For M1 we only touch the corpus.

We download with the standard library only — no heavy dependency — and cache to
``data/raw/`` so we fetch once.
"""
from __future__ import annotations

import json
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi

from .. import config
from .models import Document


def _download_and_extract(url: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_dir / "_download.zip"
    print(f"Downloading {url} ...")
    # Use certifi's CA bundle explicitly: on Windows, Python's ssl module often
    # can't find the system root certificates, so default verification fails.
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(url, headers={"User-Agent": "measurable-rag/0.1"})
    with urllib.request.urlopen(req, timeout=180, context=ctx) as resp, open(
        tmp_path, "wb"
    ) as out:
        shutil.copyfileobj(resp, out)
    print(f"Extracting into {dest_dir} ...")
    with zipfile.ZipFile(tmp_path) as zf:
        zf.extractall(dest_dir)
    tmp_path.unlink()


def load_scifact_corpus(
    raw_dir: Path | None = None, download: bool = True
) -> dict[str, Document]:
    """Load the SciFact corpus as ``{doc_id: Document}``.

    Each document's canonical ``text`` is ``title + "\\n\\n" + abstract`` so that
    chunk offsets index into one unambiguous source string.
    """
    raw_dir = raw_dir or config.RAW_DIR
    corpus_path = raw_dir / "scifact" / "corpus.jsonl"

    if not corpus_path.exists():
        if not download:
            raise FileNotFoundError(
                f"{corpus_path} not found and download=False. "
                "Run with download=True to fetch SciFact."
            )
        _download_and_extract(config.SCIFACT_URL, raw_dir)

    docs: dict[str, Document] = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = str(obj["_id"])
            title = (obj.get("title") or "").strip()
            body = (obj.get("text") or "").strip()
            text = f"{title}\n\n{body}" if title else body
            docs[doc_id] = Document(
                doc_id=doc_id,
                text=text,
                title=title,
                metadata=obj.get("metadata", {}) or {},
            )
    return docs
