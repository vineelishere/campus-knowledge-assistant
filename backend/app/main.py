import sys
import threading
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .auth import create_token, current_user, require_admin, verify_password
from .config import get_settings
from .db import connection, get_pool
from .service import ask

# Values shipped in .env.example / config.py; never acceptable in production.
DEFAULT_SECRETS = {"", "dev-only-change-me", "change-me-to-a-long-random-string"}

# Per-process failed-login throttle. Good enough for one API container; a multi-replica
# deployment should move this to Redis or the edge proxy.
_login_failures: dict[str, list[float]] = {}
_login_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    s = get_settings()
    if s.jwt_secret in DEFAULT_SECRETS:
        if s.environment.strip().lower() == "production":
            raise RuntimeError(
                "JWT_SECRET is still a default value. Set a long random string before deploying."
            )
        print(
            "warning: JWT_SECRET is a development default; set a long random value before deploying.",
            file=sys.stderr,
        )
    get_pool()
    yield


app = FastAPI(title="Campus Knowledge Assistant", lifespan=lifespan)
settings = get_settings()

allowed_origins = [
    origin.strip().rstrip("/")
    for origin in settings.allowed_origins.split(",")
    if origin.strip()
]
print("CORS allowed origins:", allowed_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginIn(BaseModel):
    email: str
    password: str


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


def _login_key(request: Request, email: str) -> str:
    client = request.client.host if request.client else "unknown"
    return f"{client}|{email.strip().lower()}"


def _check_login_throttle(key: str) -> None:
    s = get_settings()
    now = time.monotonic()
    with _login_lock:
        recent = [t for t in _login_failures.get(key, []) if now - t < s.login_window_seconds]
        _login_failures[key] = recent
        if len(recent) >= s.login_max_attempts:
            retry = max(1, int(s.login_window_seconds - (now - recent[0])) + 1)
    if len(recent) >= s.login_max_attempts:
        raise HTTPException(
            429,
            f"Too many failed login attempts. Try again in {retry} seconds.",
            headers={"Retry-After": str(retry)},
        )


def _record_login_failure(key: str) -> None:
    s = get_settings()
    now = time.monotonic()
    with _login_lock:
        _login_failures.setdefault(key, []).append(now)
        if len(_login_failures) > 1000:  # keep the map from growing without bound
            cutoff = now - s.login_window_seconds
            for stale_key in [k for k, v in _login_failures.items() if not any(t >= cutoff for t in v)]:
                del _login_failures[stale_key]


def _clear_login_failures(key: str) -> None:
    with _login_lock:
        _login_failures.pop(key, None)


@app.get("/")
def root():
    return {"service": "Campus Knowledge Assistant", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    """Readiness check: a live probe must confirm the database is actually reachable."""
    try:
        with connection() as conn:
            conn.execute("SELECT 1")
    except Exception as exc:
        raise HTTPException(503, f"database unavailable: {type(exc).__name__}")
    return {"ok": True, "database": "up"}


@app.post("/auth/login")
def login(body: LoginIn, request: Request):
    key = _login_key(request, body.email)
    _check_login_throttle(key)
    with connection() as conn:
        user = conn.execute("SELECT * FROM users WHERE email = %s", (body.email,)).fetchone()
    if not user or not verify_password(body.password, user["password_hash"]):
        _record_login_failure(key)
        raise HTTPException(401, "Wrong email or password")
    _clear_login_failures(key)
    return {"token": create_token(user), "role": user["role"], "email": user["email"]}


@app.get("/me")
def me(user: dict = Depends(current_user)):
    return {"email": user["email"], "role": user["role"], "department": user["department"]}


@app.post("/ask")
def ask_endpoint(body: AskIn, user: dict = Depends(current_user)):
    a = ask(user, body.question.strip())
    return {
        "answer": a.answer,
        "status": a.status,
        "citations": a.citations,
        "warnings": a.warnings,
        "latency_ms": a.latency_ms,
    }


@app.get("/admin/metrics")
def metrics(days: int = 7, _: dict = Depends(require_admin)):
    """Dashboard data straight from query_log (Langfuse holds the per-step traces)."""
    with connection() as conn:
        summary = conn.execute(
            """SELECT count(*) AS total,
                      count(*) FILTER (WHERE status = 'answered')   AS answered,
                      count(*) FILTER (WHERE status = 'abstained')  AS abstained,
                      count(*) FILTER (WHERE status = 'stale_only') AS stale_only,
                      count(*) FILTER (WHERE status = 'unsupported') AS unsupported,
                      count(*) FILTER (WHERE status = 'error')      AS errors,
                      coalesce(percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms), 0) AS p50_ms,
                      coalesce(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms), 0) AS p95_ms,
                      coalesce(sum(cost_usd), 0) AS total_cost_usd
               FROM query_log WHERE ts > now() - make_interval(days => %s)""",
            (days,),
        ).fetchone()
        failed = conn.execute(
            """SELECT ts, question, status, reason, top_score FROM query_log
               WHERE status <> 'answered' AND ts > now() - make_interval(days => %s)
               ORDER BY ts DESC LIMIT 100""",
            (days,),
        ).fetchall()
        daily = conn.execute(
            """SELECT date_trunc('day', ts) AS day, count(*) AS queries,
                      count(*) FILTER (WHERE status <> 'answered') AS not_answered,
                      avg(latency_ms) AS avg_ms, sum(cost_usd) AS cost_usd
               FROM query_log WHERE ts > now() - make_interval(days => %s)
               GROUP BY 1 ORDER BY 1""",
            (days,),
        ).fetchall()
    return {"summary": summary, "failed_searches": failed, "daily": daily}
