"""Load SciFact, chunk it, and print stats + a sample so you can SEE the result.

Run:  python scripts/chunk_corpus.py
"""
from __future__ import annotations

from measurable_rag import config
from measurable_rag.data.chunking import chunk_document
from measurable_rag.data.loaders import load_scifact_corpus


def main() -> None:
    docs = load_scifact_corpus()
    print(f"\nLoaded {len(docs):,} documents from SciFact.")

    # Chunk everything, keeping each doc's chunks grouped so we can demo overlap.
    per_doc: dict[str, list] = {}
    all_chunks = []
    for doc_id, doc in docs.items():
        chunks = chunk_document(doc, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        per_doc[doc_id] = chunks
        all_chunks.extend(chunks)

    lengths = [c.end - c.start for c in all_chunks]
    n_multi = sum(1 for cs in per_doc.values() if len(cs) > 1)
    print(f"Chunk size / overlap: {config.CHUNK_SIZE} / {config.CHUNK_OVERLAP} chars")
    print(f"Produced {len(all_chunks):,} chunks "
          f"({len(all_chunks) / max(len(docs), 1):.2f} per doc).")
    print(f"Chunk length (chars): min={min(lengths)} "
          f"avg={sum(lengths) / len(lengths):.0f} max={max(lengths)}")
    print(f"Docs that split into >1 chunk: {n_multi:,} / {len(docs):,}")

    # --- Correctness check: the offset invariant must hold for EVERY chunk ----
    bad = [c for c in all_chunks if not c.verify(docs[c.doc_id].text)]
    print(f"\nOffset invariant source[start:end]==text holds for "
          f"{len(all_chunks) - len(bad):,}/{len(all_chunks):,} chunks.")
    if bad:
        print(f"  !! {len(bad)} chunks FAILED the invariant — bug in the chunker.")

    # --- Show a multi-chunk document so the overlap is visible ----------------
    sample_id = next((d for d, cs in per_doc.items() if len(cs) >= 2), None)
    if sample_id is not None:
        doc = docs[sample_id]
        cs = per_doc[sample_id]
        print(f"\nSample multi-chunk document: doc_id={sample_id!r} "
              f"(title: {doc.title[:70]!r})")
        print(f"Source length: {len(doc.text)} chars -> {len(cs)} chunks")
        for c in cs:
            print(f"\n  chunk_id = {c.chunk_id}")
            print(f"  offsets  = [{c.start}, {c.end})  (len {c.end - c.start})")
            preview = c.text if len(c.text) <= 200 else c.text[:120] + " ... " + c.text[-60:]
            print(f"  text     = {preview!r}")
        # Spell out the overlap between the first two chunks.
        a, b = cs[0], cs[1]
        if b.start < a.end:
            overlap_text = doc.text[b.start:a.end]
            print(f"\n  Overlap between chunk 1 and chunk 2: "
                  f"chars [{b.start}, {a.end}) = {overlap_text!r}")


if __name__ == "__main__":
    main()
