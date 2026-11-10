"""JWT login + role-based access control.

Role rank maps directly onto documents.access_level: a user can see a document
when access_level <= rank(role). The filter is applied in SQL *before* any search.
"""
import sys
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import get_settings
from .db import connection

ROLE_RANK = {"student": 0, "professor": 1, "admin": 2}
_bearer = HTTPBearer(auto_error=False)


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def verify_password(pw: str, hashed: str) -> bool:
    return bcrypt.checkpw(pw.encode(), hashed.encode())


def create_token(user: dict) -> str:
    s = get_settings()
    payload = {
        "sub": str(user["id"]),
        "role": user["role"],
        "exp": datetime.now(timezone.utc) + timedelta(minutes=s.jwt_expire_minutes),
    }
    return jwt.encode(payload, s.jwt_secret, algorithm="HS256")


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> dict:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not logged in")
    try:
        payload = jwt.decode(creds.credentials, get_settings().jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    # Re-read the role from the DB so a demoted user loses access immediately.
    with connection() as conn:
        user = conn.execute(
            "SELECT id, email, role, department FROM users WHERE id = %s", (int(payload["sub"]),)
        ).fetchone()
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unknown user")
    user["rank"] = ROLE_RANK[user["role"]]
    return user


def require_admin(user: dict = Depends(current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Administrators only")
    return user


def create_user(email: str, password: str, role: str, department: str | None = None) -> None:
    if role not in ROLE_RANK:
        raise ValueError(f"role must be one of {list(ROLE_RANK)}")
    with connection() as conn:
        conn.execute(
            "INSERT INTO users (email, password_hash, role, department) VALUES (%s,%s,%s,%s)",
            (email, hash_password(password), role, department),
        )
        conn.commit()


if __name__ == "__main__":
    # python -m app.auth email password role [department]
    if len(sys.argv) < 4:
        sys.exit("usage: python -m app.auth EMAIL PASSWORD ROLE [DEPARTMENT]")
    create_user(*sys.argv[1:5])
    print("created", sys.argv[1])
