"""FastAPI backend for Kira — accounts (JWT), SQLite persistence, multi-user.

Auth-gated: every request needs a bearer token (except register/login). Chats
and uploaded-document collections are stored server-side per user. The /api/ask
endpoint persists both the user message and Kira's answer.

Run:  uvicorn api.main:app --port 8000
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from measurable_rag import config
from measurable_rag.data.user_docs import load_documents
from measurable_rag.generation.abstain import grounding_score, should_abstain
from measurable_rag.generation.generator import Generator
from measurable_rag.generation.verify import Verifier, citation_precision, faithfulness
from measurable_rag.retrieval.build import build_retriever
from measurable_rag.retrieval.dense import DenseIndex
from measurable_rag.retrieval.embedder import Embedder
from measurable_rag.retrieval.pipeline import HybridRetriever
from measurable_rag.retrieval.rerank import Reranker
from measurable_rag.retrieval.sparse import SparseIndex
from measurable_rag.retrieval.web import search_web

from .auth import current_user, hash_pw, make_token, verify_pw
from .db import Collection, Message, Thread, User, get_db, init_db, now

app = FastAPI(title="Kira API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
init_db()

COLLECTIONS_DIR = config.DATA_DIR / "collections"
_state: dict = {"embedder": None, "scifact": None, "reranker": None}
_collections: dict[str, HybridRetriever] = {}  # cache of on-disk doc indexes


def embedder() -> Embedder:
    if _state["embedder"] is None:
        _state["embedder"] = Embedder()
    return _state["embedder"]


def scifact_indexes():
    if _state["scifact"] is None:
        if not (config.INDEX_DIR / "dense.faiss").exists():
            # The benchmark index isn't shipped in deployments — guide the user
            # to the features that work (their own docs / web) instead of 500ing.
            raise HTTPException(503, "The SciFact demo corpus isn't available here. "
                                     "Upload your own documents or use web search.")
        index = DenseIndex.load(config.INDEX_DIR)
        _state["scifact"] = (index, SparseIndex(index.chunks))
    return _state["scifact"]


def reranker() -> Reranker:
    if _state["reranker"] is None:
        _state["reranker"] = Reranker()
    return _state["reranker"]


def load_collection(cid: str) -> HybridRetriever:
    """Load a persisted uploaded-doc collection from disk (cached)."""
    if cid not in _collections:
        index = DenseIndex.load(COLLECTIONS_DIR / cid)
        _collections[cid] = HybridRetriever(index, SparseIndex(index.chunks), embedder(), None)
    return _collections[cid]


def _groq_http_error(e: Exception) -> HTTPException:
    name, msg = type(e).__name__, str(e)
    if "Authentication" in name or "invalid_api_key" in msg or "401" in msg:
        return HTTPException(401, "Groq rejected the API key (set a valid GROQ_API_KEY in .env).")
    if "RateLimit" in name or "429" in msg:
        return HTTPException(429, "Groq rate limit reached — wait a few seconds and try again.")
    return HTTPException(502, f"Generation failed: {msg[:200]}")


@app.get("/api/health")
def health():
    return {"ok": True}


# ---------------------------------------------------------------- auth
class Creds(BaseModel):
    email: str
    password: str


@app.post("/api/auth/register")
@limiter.limit("10/minute")
def register(request: Request, body: Creds, db: Session = Depends(get_db)):
    email = body.email.strip().lower()
    if not email or not body.password:
        raise HTTPException(400, "Email and password are required.")
    if db.query(User).filter_by(email=email).first():
        raise HTTPException(409, "That email is already registered.")
    user = User(id=uuid.uuid4().hex, email=email, password_hash=hash_pw(body.password), created_at=now())
    db.add(user)
    db.commit()
    return {"token": make_token(user.id), "email": user.email}


@app.post("/api/auth/login")
@limiter.limit("10/minute")
def login(request: Request, body: Creds, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(email=body.email.strip().lower()).first()
    if not user or not verify_pw(body.password, user.password_hash):
        raise HTTPException(401, "Invalid email or password.")
    return {"token": make_token(user.id), "email": user.email}


@app.get("/api/me")
def me(user: User = Depends(current_user)):
    return {"email": user.email}


# ---------------------------------------------------------------- threads
def serialize_thread(db: Session, t: Thread) -> dict:
    msgs = db.query(Message).filter_by(thread_id=t.id).order_by(Message.created_at).all()
    return {
        "id": t.id, "title": t.title, "docLabel": t.doc_label, "hasDocs": bool(t.collection_id),
        "messages": [
            {"id": m.id, "role": m.role, "content": m.content,
             "result": json.loads(m.result_json) if m.result_json else None,
             "images": json.loads(m.images_json) if m.images_json else None,
             "ts": m.created_at}
            for m in msgs
        ],
    }


@app.get("/api/threads")
def list_threads(user: User = Depends(current_user), db: Session = Depends(get_db)):
    threads = db.query(Thread).filter_by(user_id=user.id).order_by(Thread.updated_at.desc()).all()
    return [serialize_thread(db, t) for t in threads]


class NewThread(BaseModel):
    title: str | None = None


@app.post("/api/threads")
def create_thread(body: NewThread, user: User = Depends(current_user), db: Session = Depends(get_db)):
    t = Thread(id=uuid.uuid4().hex, user_id=user.id, title=(body.title or "New chat"),
               created_at=now(), updated_at=now())
    db.add(t)
    db.commit()
    return serialize_thread(db, t)


class RenameThread(BaseModel):
    title: str


def _owned_thread(tid: str, user: User, db: Session) -> Thread:
    t = db.get(Thread, tid)
    if not t or t.user_id != user.id:
        raise HTTPException(404, "Thread not found")
    return t


@app.patch("/api/threads/{tid}")
def rename_thread(tid: str, body: RenameThread, user: User = Depends(current_user), db: Session = Depends(get_db)):
    t = _owned_thread(tid, user, db)
    t.title = body.title.strip() or t.title
    db.commit()
    return {"ok": True}


@app.delete("/api/threads/{tid}")
def delete_thread(tid: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    t = _owned_thread(tid, user, db)
    db.query(Message).filter_by(thread_id=tid).delete()
    db.delete(t)
    db.commit()
    return {"ok": True}


# ---------------------------------------------------------------- upload
@app.post("/api/upload")
@limiter.limit("15/minute")
async def upload(request: Request, thread_id: str = Form(...), files: list[UploadFile] = File(...),
                 user: User = Depends(current_user), db: Session = Depends(get_db)):
    t = _owned_thread(thread_id, user, db)
    payloads = [(f.filename or "file", await f.read()) for f in files]
    docs = load_documents(payloads)
    if not docs:
        raise HTTPException(400, "No text could be extracted from those files.")
    rtr, n_chunks = build_retriever(docs, embedder(), None)
    cid = uuid.uuid4().hex
    rtr.dense_index.save(COLLECTIONS_DIR / cid)  # persist so it survives restarts
    _collections[cid] = rtr
    label = f"{len(docs)} doc(s) · {n_chunks} chunks"
    db.add(Collection(id=cid, user_id=user.id, label=label, created_at=now()))
    t.collection_id = cid
    t.doc_label = label
    db.commit()
    return {"collection_id": cid, "n_docs": len(docs), "n_chunks": n_chunks, "label": label}


# ---------------------------------------------------------------- ask
class AskBody(BaseModel):
    thread_id: str
    query: str
    use_rag: bool = False
    use_web: bool = False
    kb: str = "docs"
    mode: str = "hybrid"
    top_k: int = 5
    web_results: int = 5
    threshold: float = 0.5
    images: list[str] = []


def _retrieve(rtr: HybridRetriever, body: AskBody):
    if body.mode == "dense":
        return rtr.dense(body.query, body.top_k)
    if body.mode == "rerank":
        return rtr.hybrid_reranked(body.query, body.top_k)
    return rtr.hybrid(body.query, body.top_k)


def _generate(body: AskBody, thread: Thread, gen: Generator, history: list[dict], key: str) -> dict:
    # Vision: answer about attached image(s), ungrounded.
    if body.images:
        try:
            answer = gen.answer_vision(body.query, body.images, history)
        except Exception as e:
            raise _groq_http_error(e)
        return {"answer": answer, "grounded": False, "abstained": False, "refused_by_model": False,
                "grounding": None, "citation_precision": None, "faithfulness": None,
                "verdicts": [], "sources": [], "mode": "vision"}

    rr = reranker() if body.mode == "rerank" else None
    tagged: list[tuple[object, str]] = []
    if body.use_rag:
        if body.kb == "scifact":
            index, sparse = scifact_indexes()
            rtr = HybridRetriever(index, sparse, embedder(), rr)
            tagged += [(c, "scifact") for c in _retrieve(rtr, body)]
        elif thread.collection_id:
            base = load_collection(thread.collection_id)
            rtr = HybridRetriever(base.dense_index, base.sparse_index, embedder(), rr)
            tagged += [(c, "doc") for c in _retrieve(rtr, body)]
    if body.use_web:
        web_docs = search_web(body.query, max_results=body.web_results)
        if web_docs:
            rtr, _ = build_retriever(web_docs, embedder(), rr)
            tagged += [(c, "web") for c in _retrieve(rtr, body)]

    seen, per_doc, chunks, origins = set(), {}, [], []
    for c, o in tagged:
        if c.chunk_id in seen or per_doc.get(c.doc_id, 0) >= 2:
            continue
        seen.add(c.chunk_id)
        per_doc[c.doc_id] = per_doc.get(c.doc_id, 0) + 1
        chunks.append(c)
        origins.append(o)

    try:
        if not chunks:
            return {"answer": gen.answer_plain(body.query, history), "grounded": False,
                    "abstained": False, "refused_by_model": False, "grounding": None,
                    "citation_precision": None, "faithfulness": None, "verdicts": [],
                    "sources": [], "mode": "ungrounded"}
        answer = gen.answer(body.query, chunks, history)
        refused = answer.strip().startswith("INSUFFICIENT")
        verdicts = [] if refused else Verifier(api_key=key).verify(answer, chunks)
    except Exception as e:
        raise _groq_http_error(e)

    quotes: dict[int, list[str]] = {}
    for v in verdicts:
        if v.supported and v.quote:
            quotes.setdefault(v.source_num, []).append(v.quote)
    return {
        "answer": answer, "grounded": True, "abstained": should_abstain(answer, verdicts, body.threshold),
        "refused_by_model": refused, "grounding": grounding_score(verdicts),
        "citation_precision": citation_precision(verdicts), "faithfulness": faithfulness(verdicts),
        "verdicts": [{"source_num": v.source_num, "doc_id": v.doc_id, "supported": v.supported,
                      "claim": v.claim, "quote": v.quote} for v in verdicts],
        "sources": [{"n": i + 1, "doc_id": c.doc_id, "text": c.text,
                     "quotes": quotes.get(i + 1, []), "origin": origins[i]}
                    for i, c in enumerate(chunks)],
        "mode": "grounded",
    }


@app.post("/api/ask")
@limiter.limit("30/minute")
def ask(request: Request, body: AskBody, user: User = Depends(current_user), db: Session = Depends(get_db)):
    t = _owned_thread(body.thread_id, user, db)
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise HTTPException(500, "Server has no GROQ_API_KEY configured (.env).")
    gen = Generator(api_key=key)

    prior = db.query(Message).filter_by(thread_id=t.id).order_by(Message.created_at).all()
    history = [{"role": m.role, "content": m.content} for m in prior]

    result = _generate(body, t, gen, history, key)  # raises HTTPException on failure (nothing persisted)

    db.add(Message(id=uuid.uuid4().hex, thread_id=t.id, role="user", content=body.query,
                   images_json=json.dumps(body.images) if body.images else None, created_at=now()))
    db.add(Message(id=uuid.uuid4().hex, thread_id=t.id, role="assistant", content=result["answer"],
                   result_json=json.dumps(result), created_at=now() + 0.001))
    if t.title == "New chat":
        t.title = body.query[:42]
    t.updated_at = now()
    db.commit()
    return result


_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")
