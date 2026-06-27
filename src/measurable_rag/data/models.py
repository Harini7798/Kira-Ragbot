"""Core data types: Document and Chunk.

A ``Document`` is one source text. A ``Chunk`` is a contiguous slice of that
text, tagged with the exact character offsets it came from.

Those offsets are the foundation of everything later in the project: a
span-level citation is *just* a chunk's ``(doc_id, start, end)``, and the UI can
highlight ``source[start:end]`` to show exactly which characters a claim rests
on. So the one invariant we never break is:

    source_text[chunk.start : chunk.end] == chunk.text
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """One source text in the corpus.

    ``text`` is the canonical string that chunk offsets index into. For SciFact
    we build it as ``title + "\\n\\n" + abstract`` so a document has a single
    unambiguous source string; ``title`` is also kept separately for display.
    """

    doc_id: str
    text: str
    title: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a Document's text.

    Frozen (immutable + hashable) on purpose: chunks get used as dict keys and
    set members when computing retrieval metrics, so they must be stable.
    """

    chunk_id: str
    doc_id: str
    text: str
    start: int  # inclusive char offset into the source Document.text
    end: int    # exclusive char offset into the source Document.text

    def verify(self, source_text: str) -> bool:
        """The invariant that makes citations exact: this chunk really is the
        slice of the source it claims to be."""
        return source_text[self.start : self.end] == self.text
