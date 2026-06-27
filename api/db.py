"""SQLite persistence (SQLAlchemy): users, chat threads, messages, doc collections.

Replaces the browser-localStorage store with a real server-side database, so
chats persist across devices/restarts and the app is genuinely multi-user
(every row is scoped to a user id).
"""
import time
from pathlib import Path

from sqlalchemy import Float, ForeignKey, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from measurable_rag import config

DB_PATH: Path = config.DATA_DIR / "kira.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# check_same_thread=False: FastAPI may use the connection across threads.
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def now() -> float:
    return time.time()


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String)
    created_at: Mapped[float] = mapped_column(Float, default=now)


class Thread(Base):
    __tablename__ = "threads"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String, default="New chat")
    collection_id: Mapped[str | None] = mapped_column(String, nullable=True)
    doc_label: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[float] = mapped_column(Float, default=now)
    updated_at: Mapped[float] = mapped_column(Float, default=now)


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    thread_id: Mapped[str] = mapped_column(String, ForeignKey("threads.id"), index=True)
    role: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    images_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[float] = mapped_column(Float, default=now)


class Collection(Base):
    """An uploaded-document set; its FAISS index lives on disk at
    data/collections/<id>/ so it survives restarts."""
    __tablename__ = "collections"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    label: Mapped[str] = mapped_column(String)
    created_at: Mapped[float] = mapped_column(Float, default=now)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
