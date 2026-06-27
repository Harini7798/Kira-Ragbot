"""Live web search as a corpus.

Given a question, search the web, fetch/clean the top results, and return them as
Document objects so the same RAG pipeline runs over fresh web content with
citations back to the source URLs.

Two backends:
  * Tavily (https://tavily.com) — a search API built FOR RAG: relevant, deduped,
    pre-extracted content. Used automatically if TAVILY_API_KEY is set (free tier
    available). This is the big quality lever.
  * DuckDuckGo (no key) — the fallback. Free but lower quality, so we at least
    enforce ONE result per domain to avoid five chunks of the same SEO page.

Resilient: any failure falls back or is skipped rather than crashing.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

from ..data.models import Document

_MAX_CHARS = 20_000


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return url


def _fetch_extract(url: str, timeout: int) -> str:
    try:
        import requests
        import trafilatura

        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if resp.ok and resp.text:
            return trafilatura.extract(resp.text, include_comments=False) or ""
    except Exception:
        return ""
    return ""


def _search_tavily(query: str, max_results: int) -> dict[str, Document]:
    import requests

    resp = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": os.environ.get("TAVILY_API_KEY"),
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
            "include_raw_content": True,
        },
        timeout=25,
    )
    resp.raise_for_status()
    docs: dict[str, Document] = {}
    for r in resp.json().get("results", []):
        url, title = r.get("url"), (r.get("title") or "").strip()
        body = (r.get("raw_content") or r.get("content") or "").strip()
        if url and body:
            text = (f"{title}\n\n{body}" if title else body)[:_MAX_CHARS]
            docs[url] = Document(doc_id=url, text=text, title=title)
    return docs


def _search_ddg(query: str, max_results: int, timeout: int) -> dict[str, Document]:
    try:
        from ddgs import DDGS

        # Over-fetch so we can keep only the first hit per domain (diversity).
        results = list(DDGS().text(query, max_results=max_results * 3))
    except Exception:
        return {}

    docs: dict[str, Document] = {}
    seen_domains: set[str] = set()
    for r in results:
        url = r.get("href") or r.get("url") or ""
        if not url or _domain(url) in seen_domains:
            continue
        seen_domains.add(_domain(url))
        title = (r.get("title") or "").strip()
        snippet = (r.get("body") or "").strip()
        body = _fetch_extract(url, timeout) or snippet  # fall back to the snippet
        if not body:
            continue
        text = (f"{title}\n\n{body}" if title else body)[:_MAX_CHARS]
        docs[url] = Document(doc_id=url, text=text, title=title)
        if len(docs) >= max_results:
            break
    return docs


def search_web(query: str, max_results: int = 5, timeout: int = 6) -> dict[str, Document]:
    """Return {url: Document} for the top web results. Prefers Tavily if configured."""
    if os.environ.get("TAVILY_API_KEY"):
        try:
            docs = _search_tavily(query, max_results)
            if docs:
                return docs
        except Exception:
            pass  # fall through to DuckDuckGo
    return _search_ddg(query, max_results, timeout)
