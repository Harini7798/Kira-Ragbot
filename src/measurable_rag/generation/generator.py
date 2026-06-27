"""Answer generation for Kira.

Two modes:
  * grounded (``answer``) — answer using ONLY the numbered retrieved sources and
    cite each claim; flag insufficiency. Citations are checked later (verify.py).
  * ungrounded (``answer_plain``) — answer from the model's own knowledge with no
    sources. Used when both RAG and web search are off; such answers are clearly
    marked as unverified in the UI.

Both accept recent chat ``history`` so Kira holds a multi-turn conversation.
The Groq key is read from the GROQ_API_KEY env var or passed in; never stored.
"""
from __future__ import annotations

import os

from .. import config
from ..data.models import Chunk

INSUFFICIENT = "INSUFFICIENT: the provided sources do not answer this question."

_SYSTEM = (
    "You are Kira, a careful assistant. Answer the question using ONLY the "
    "numbered sources provided. After each claim, cite the source number(s) it "
    "rests on in square brackets, e.g. [1] or [2][3]. If the sources do not "
    f"contain enough information to answer, reply with exactly: '{INSUFFICIENT}' "
    "Never use outside knowledge."
)

_SYSTEM_PLAIN = (
    "You are Kira, a helpful assistant. Answer the user's question clearly and "
    "concisely from your own knowledge. If you are unsure, say so honestly. "
    "Do not fabricate facts, figures, or citations."
)


def format_sources(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[{i}] (doc {c.doc_id}) {c.text}" for i, c in enumerate(chunks, start=1)
    )


def _history_messages(history: list[dict] | None, limit: int = 6) -> list[dict]:
    if not history:
        return []
    out = []
    for turn in history[-limit:]:
        role, content = turn.get("role"), turn.get("content", "")
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out


class Generator:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        from groq import Groq

        self.model = model or config.GROQ_MODEL
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set (pass it in or set the env var).")
        self.client = Groq(api_key=key)

    def _chat(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.0
        )
        return resp.choices[0].message.content.strip()

    def answer(self, query: str, chunks: list[Chunk], history: list[dict] | None = None) -> str:
        """Grounded answer with inline [n] citations into ``chunks``."""
        messages = [{"role": "system", "content": _SYSTEM}, *_history_messages(history)]
        messages.append(
            {"role": "user", "content": f"Question: {query}\n\nSources:\n{format_sources(chunks)}\n\nAnswer:"}
        )
        return self._chat(messages)

    def answer_plain(self, query: str, history: list[dict] | None = None) -> str:
        """Ungrounded answer from the model's own knowledge (no sources)."""
        messages = [{"role": "system", "content": _SYSTEM_PLAIN}, *_history_messages(history)]
        messages.append({"role": "user", "content": query})
        return self._chat(messages)

    def answer_vision(
        self, query: str, images: list[str], history: list[dict] | None = None
    ) -> str:
        """Answer a question about one or more images (base64 data URLs), using
        the multimodal model. Ungrounded — the answer is about the image."""
        content: list[dict] = [{"type": "text", "text": query}]
        for url in images:
            content.append({"type": "image_url", "image_url": {"url": url}})
        messages = [
            {"role": "system", "content": _SYSTEM_PLAIN},
            *_history_messages(history),
            {"role": "user", "content": content},
        ]
        resp = self.client.chat.completions.create(
            model=config.VISION_MODEL, messages=messages, temperature=0.2
        )
        return resp.choices[0].message.content.strip()
