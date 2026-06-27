"""Email/password auth: bcrypt password hashing + JWT bearer tokens.

Self-contained — no third-party auth provider. The token signing secret is read
from KIRA_SECRET (set it in .env); a dev fallback is used if unset.
"""
import os
import time

import bcrypt
import jwt
from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from .db import User, get_db

SECRET = os.environ.get("KIRA_SECRET", "kira-dev-secret-change-me")
ALGO = "HS256"


def hash_pw(password: str) -> str:
    # bcrypt caps at 72 bytes; slice to stay within it.
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_pw(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:72], hashed.encode("utf-8"))
    except Exception:
        return False


def make_token(user_id: str) -> str:
    return jwt.encode({"sub": user_id, "iat": int(time.time())}, SECRET, algorithm=ALGO)


def decode_token(token: str) -> str:
    return jwt.decode(token, SECRET, algorithms=[ALGO])["sub"]


def current_user(
    authorization: str | None = Header(None), db: Session = Depends(get_db)
) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Not authenticated")
    try:
        user_id = decode_token(authorization.split(" ", 1)[1])
    except Exception:
        raise HTTPException(401, "Invalid or expired session")
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(401, "User not found")
    return user
