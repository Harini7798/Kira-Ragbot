"""Citation verification — do the cited sources actually support each claim?

LLMs cite confidently and sometimes wrongly (in our demo the generator cited a
paper about mulberry leaves to back a claim about cardiovascular disease). So we
never trust citations; we verify them with a SEPARATE pass that is deliberately
independent of the generator in two ways:

  1. A *different* model does the judging (config.VERIFIER_MODEL != GROQ_MODEL),
     so a model isn't grading its own work — the evaluator-bias pitfall.
  2. The judge must return the exact sentence from the source that supports the
     claim, and we then confirm that sentence literally occurs in the cited
     chunk (offset check). If it doesn't, the citation is rejected — so the judge
     can't rescue a bad citation by inventing a justification either.

From the verdicts we compute the two headline citation numbers:
  * citation precision — of all (claim -> cited source) links, the fraction that
    genuinely support the claim.
  * faithfulness — the fraction of the answer's claims grounded in >=1 source.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from .. import config
from ..data.models import Chunk

_CITE = re.compile(r"\[(\d+)\]")
_HAS_LETTER = re.compile(r"[A-Za-z]")
# Split an ANSWER into sentences locally — independent of the corpus chunker, so
# tweaking this never changes chunk boundaries / invalidates the built index.
# Breaks after . ! ? followed by whitespace, so a trailing citation-only cluster
# (e.g. "[1][3][4][5]") becomes its own segment and is dropped below.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class CitationVerdict:
    claim: str
    source_num: int              # which numbered source [n] this checks
    doc_id: str
    supported: bool              # source entails the claim AND quote was verified
    quote: str = ""              # exact supporting sentence found in the source
    span: tuple[int, int] | None = None  # (start, end) offsets into the doc text


def split_claims(answer: str) -> list[tuple[str, list[int]]]:
    """Break an answer into (claim_text, cited_source_numbers) per sentence.

    The [n] markers in a sentence are its citations; the claim text is the
    sentence with those markers stripped. Sentences with no letters (e.g. a bare
    "[1][3]." trailer) are dropped.
    """
    claims: list[tuple[str, list[int]]] = []
    for sentence in _SENT_SPLIT.split(answer.strip()):
        cited = [int(m.group(1)) for m in _CITE.finditer(sentence)]
        text = _CITE.sub("", sentence)
        text = re.sub(r"\s+", " ", text)                      # collapse whitespace
        text = re.sub(r"\s+([.,;:!?])", r"\1", text).strip()  # drop space before punctuation
        if _HAS_LETTER.search(text):
            claims.append((text, cited))
    return claims


def citation_precision(verdicts: list[CitationVerdict]) -> float | None:
    """Fraction of citation links that genuinely support their claim.

    None if the answer made no citations at all (precision is undefined).
    """
    if not verdicts:
        return None
    return sum(1 for v in verdicts if v.supported) / len(verdicts)


def faithfulness(verdicts: list[CitationVerdict]) -> float:
    """Fraction of cited claims that are supported by at least one cited source."""
    by_claim: dict[str, bool] = {}
    for v in verdicts:
        by_claim[v.claim] = by_claim.get(v.claim, False) or v.supported
    if not by_claim:
        return 0.0
    return sum(by_claim.values()) / len(by_claim)


_JUDGE_SYSTEM = (
    "You are a strict scientific fact-checker. Decide whether the SOURCE directly "
    "supports the CLAIM. Respond ONLY as JSON: "
    '{"supported": <true|false>, "quote": "<the exact sentence copied verbatim '
    'from the SOURCE that supports the claim, or empty string>"}. '
    "Mark supported true only if the source explicitly supports the claim; if the "
    "source is unrelated or merely on a similar topic, mark it false."
)


class Verifier:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        from groq import Groq

        self.model = model or config.VERIFIER_MODEL
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set (set it in your shell).")
        self.client = Groq(api_key=key)

    def _judge(self, claim: str, source_text: str) -> tuple[bool, str]:
        user = f"CLAIM: {claim}\n\nSOURCE:\n{source_text}"
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            return bool(data.get("supported", False)), str(data.get("quote", "")).strip()
        except (json.JSONDecodeError, KeyError, TypeError):
            return False, ""  # if the judge misbehaves, fail closed (unsupported)

    def verify(
        self, answer: str, chunks: list[Chunk], max_cites_per_claim: int = 4
    ) -> list[CitationVerdict]:
        """Verify every (claim -> cited source) link in the answer."""
        verdicts: list[CitationVerdict] = []
        for claim, cited in split_claims(answer):
            for n in cited[:max_cites_per_claim]:
                if not (1 <= n <= len(chunks)):
                    continue  # citation points to a source that doesn't exist
                chunk = chunks[n - 1]
                supported, quote = self._judge(claim, chunk.text)
                span = None
                # The quote must literally appear in the cited chunk; otherwise the
                # judge hallucinated its justification and we reject the citation.
                if quote and quote in chunk.text:
                    idx = chunk.text.index(quote)
                    span = (chunk.start + idx, chunk.start + idx + len(quote))
                elif supported:
                    supported = False
                    quote = ""
                verdicts.append(
                    CitationVerdict(claim, n, chunk.doc_id, supported, quote, span)
                )
        return verdicts
