"""Sentence-aware chunking with stable IDs and exact character offsets.

Three design goals, each tied to a requirement of the wider project:

* **Exact offsets** — ``chunk.text`` is always ``source[start:end]``, verbatim.
  This is what lets a citation point to (and a UI highlight) the precise span a
  claim rests on. We get it for free by *slicing* the source rather than
  stitching sentences back together.

* **Stable IDs** — a chunk's ID is derived from its document and its offsets, so
  re-running chunking with the same settings yields byte-identical IDs. Metrics
  and citations reference chunks by ID across many runs; if IDs drifted, the
  numbers would be meaningless.

* **Overlap** — adjacent chunks deliberately share a little text. A fact sitting
  near a boundary then lands *whole* inside at least one chunk, instead of being
  split across two so that neither alone supports it. (That split is one of the
  pitfalls this project is meant to design against.)

The sentence splitter below is intentionally simple — a regex. Crude sentence
boundaries only hurt chunk *quality*, never *correctness*: because every chunk's
text is a literal slice of the source, the offset invariant holds no matter
where we cut. We can swap in a smarter splitter later without changing anything
downstream.
"""
from __future__ import annotations

import re

from .models import Chunk, Document

# Split at whitespace that follows sentence-ending punctuation and precedes the
# likely start of a new sentence (capital letter, digit, or opening paren).
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def _strip_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Trim leading/trailing whitespace from a [start, end) span."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Char spans of the sentences in ``text``.

    Each ``(start, end)`` is such that ``text[start:end]`` is one sentence with
    surrounding whitespace trimmed. Returning spans (not strings) is the whole
    point — it keeps us anchored to absolute positions in the source.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _SENTENCE_BOUNDARY.finditer(text):
        s, e = _strip_span(text, pos, m.start())
        if e > s:
            spans.append((s, e))
        pos = m.end()
    s, e = _strip_span(text, pos, len(text))
    if e > s:
        spans.append((s, e))
    return spans


def _next_index(spans: list[tuple[int, int]], i: int, j: int, overlap: int) -> int:
    """Index where the *next* chunk should start.

    The current chunk covers sentences ``[i, j)``. To keep a fact near the
    boundary whole inside a neighbour, the next chunk re-includes trailing
    sentences of this one:

    * ``overlap <= 0`` — no overlap; the next chunk starts right after this one.
    * ``overlap > 0``  — re-include the last sentence *always* (so there is real
      overlap, even when that one sentence is longer than ``overlap``), then keep
      re-including earlier sentences while the overlap stays within ``overlap``
      characters.

    So overlap is rounded to whole sentences: at least one when requested. This
    is the deliberate fix for the subtle bug where, if every sentence happened to
    be longer than ``overlap``, the chunker produced *zero* overlap and silently
    defeated its own purpose.

    Forward progress is guaranteed: the returned index is always > ``i``.
    """
    if j >= len(spans) or overlap <= 0:
        return j
    end_char = spans[j - 1][1]
    # At least the last sentence — but only if that still advances past i.
    next_i = j - 1 if (j - 1) > i else j
    k = next_i - 1
    while k > i and end_char - spans[k][0] <= overlap:
        next_i = k
        k -= 1
    return next_i


def _make_chunk(doc: Document, start: int, end: int) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}::{start}-{end}",  # stable: derived from offsets
        doc_id=doc.doc_id,
        text=doc.text[start:end],
        start=start,
        end=end,
    )


def chunk_document(doc: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk one document into overlapping, sentence-aligned spans."""
    spans = sentence_spans(doc.text)
    if not spans:
        return []

    chunks: list[Chunk] = []
    i = 0
    while i < len(spans):
        start = spans[i][0]
        # Greedily pack whole sentences until adding the next would exceed the
        # size budget. Always take at least one sentence (so an over-long
        # sentence becomes its own slightly-too-big chunk rather than nothing).
        j = i + 1
        while j < len(spans) and spans[j][1] - start <= chunk_size:
            j += 1
        end = spans[j - 1][1]
        chunks.append(_make_chunk(doc, start, end))
        if j >= len(spans):
            break
        i = _next_index(spans, i, j, overlap)
    return chunks


def chunk_corpus(
    docs: dict[str, Document], chunk_size: int, overlap: int
) -> list[Chunk]:
    """Chunk every document in a corpus into a flat list of chunks."""
    chunks: list[Chunk] = []
    for doc in docs.values():
        chunks.extend(chunk_document(doc, chunk_size, overlap))
    return chunks
