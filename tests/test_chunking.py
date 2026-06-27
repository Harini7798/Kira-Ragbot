"""Tests for the chunker — the foundation that span citations depend on.

These all PASS: chunking is plumbing I build for you. (The metric/abstention
code that's yours to write will instead come as stubs with *failing* tests.)
"""
from __future__ import annotations

from measurable_rag.data.chunking import (
    chunk_document,
    sentence_spans,
)
from measurable_rag.data.models import Document

# A multi-sentence text long enough to span several chunks at small sizes.
PARAGRAPH = (
    "Vaccines train the immune system to recognize pathogens. "
    "They do this by presenting a harmless piece of the pathogen. "
    "The body then produces antibodies in response. "
    "These antibodies persist and provide future protection. "
    "Booster doses can renew waning immunity over time."
)


def make_doc(text: str, doc_id: str = "d1") -> Document:
    return Document(doc_id=doc_id, text=text, title="")


def test_offset_invariant_holds_for_every_chunk():
    """The core guarantee: source[start:end] == chunk.text, always."""
    doc = make_doc(PARAGRAPH)
    chunks = chunk_document(doc, chunk_size=120, overlap=40)
    assert chunks, "expected at least one chunk"
    for c in chunks:
        assert doc.text[c.start : c.end] == c.text
        assert c.verify(doc.text)


def test_chunk_ids_are_stable_across_runs():
    """Re-chunking the same doc with the same settings yields identical IDs."""
    doc = make_doc(PARAGRAPH)
    ids_a = [c.chunk_id for c in chunk_document(doc, 120, 40)]
    ids_b = [c.chunk_id for c in chunk_document(doc, 120, 40)]
    assert ids_a == ids_b
    assert len(set(ids_a)) == len(ids_a), "chunk IDs must be unique within a doc"


def test_adjacent_chunks_overlap():
    """With overlap > 0 and multiple chunks, each chunk starts before the
    previous one ended — so a fact near a boundary survives whole somewhere."""
    doc = make_doc(PARAGRAPH)
    chunks = chunk_document(doc, chunk_size=120, overlap=40)
    assert len(chunks) >= 2, "test text should split into multiple chunks"
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.start < prev.end, "expected overlap between adjacent chunks"


def test_chunks_cover_whole_document():
    """No source content is dropped: first chunk starts at the first sentence,
    last chunk ends at the last sentence."""
    doc = make_doc(PARAGRAPH)
    spans = sentence_spans(doc.text)
    chunks = chunk_document(doc, chunk_size=120, overlap=40)
    assert chunks[0].start == spans[0][0]
    assert chunks[-1].end == spans[-1][1]


def test_short_text_is_one_chunk():
    doc = make_doc("A single short sentence.")
    chunks = chunk_document(doc, chunk_size=800, overlap=150)
    assert len(chunks) == 1
    assert chunks[0].text == "A single short sentence."


def test_empty_text_yields_no_chunks():
    assert chunk_document(make_doc("   "), 800, 150) == []
    assert chunk_document(make_doc(""), 800, 150) == []


def test_long_sentence_becomes_its_own_chunk():
    """A single sentence longer than chunk_size must still produce a chunk
    (we never silently drop content), and the invariant must still hold."""
    long_sentence = "word " * 300 + "end."  # ~1500 chars, one sentence
    doc = make_doc(long_sentence)
    chunks = chunk_document(doc, chunk_size=200, overlap=40)
    assert len(chunks) == 1
    assert chunks[0].verify(doc.text)


def test_no_chunk_exceeds_size_unless_single_sentence():
    """Multi-sentence packing should respect the budget; only an individually
    over-long sentence is allowed to exceed it."""
    doc = make_doc(PARAGRAPH)
    size = 120
    for c in chunk_document(doc, chunk_size=size, overlap=30):
        n_sentences = len(sentence_spans(c.text))
        if n_sentences > 1:
            assert c.end - c.start <= size
