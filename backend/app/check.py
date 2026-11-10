"""Self-check for a fresh install. Run from the backend/ folder with the venv active:

    python -m app.check

Each step prints PASS / WARN / FAIL with a hint. Later steps still run after a failure,
so you see every problem at once.
"""
import sys

from .config import get_settings

results: list[tuple[str, str]] = []


def report(status: str, name: str, detail: str = "") -> None:
    results.append((status, name))
    print(f"[{status}] {name}" + (f"\n       {detail}" if detail else ""))


def check_python() -> None:
    v = sys.version_info
    if (3, 11) <= v[:2] <= (3, 12):
        report("PASS", f"Python {v.major}.{v.minor}")
    else:
        report("WARN", f"Python {v.major}.{v.minor}", "3.11 or 3.12 is recommended; Docling/torch may not support others.")


def check_env() -> None:
    s = get_settings()
    # HF_TOKEN is only required when an embedding or answer backend actually calls Hugging Face.
    needs_hf = s.embed_backend == "hf_api" or not s.llm_base_url
    if s.hf_token:
        report("PASS", "HF_TOKEN is set")
    elif needs_hf:
        report("FAIL", "HF_TOKEN is empty but a Hugging Face backend is configured",
               "Set HF_TOKEN in the project-root .env, or use local backends: EMBED_BACKEND=local and an "
               "OpenAI-compatible LLM_BASE_URL (Gemini/Ollama/Groq).")
    else:
        report("PASS", "HF_TOKEN not needed (local embeddings + OpenAI-compatible LLM)")
    if s.jwt_secret in ("", "dev-only-change-me", "change-me-to-a-long-random-string"):
        report("WARN", "JWT_SECRET is a default value",
               "Fine locally; the API refuses to start with it when ENVIRONMENT=production.")
    else:
        report("PASS", "JWT_SECRET is set")


def check_database() -> None:
    s = get_settings()
    try:
        import psycopg

        with psycopg.connect(s.database_url, connect_timeout=5) as conn:
            conn.execute("SELECT 1")
            report("PASS", "Connected to Postgres")

            ext = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
            if ext:
                report("PASS", "pgvector extension installed")
            else:
                report("FAIL", "pgvector extension missing",
                       "Use the pgvector/pgvector:pg16 image (docker compose up -d db), or run: CREATE EXTENSION vector;")

            tables = {r[0] for r in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'").fetchall()}
            missing = {"users", "documents", "chunks", "query_log"} - tables
            if missing:
                report("FAIL", f"Tables missing: {sorted(missing)}",
                       "Schema only auto-applies on a brand-new volume. Run: docker compose down -v && docker compose up -d db "
                       "(this wipes the database), or apply backend/app/schema.sql manually.")
            else:
                counts = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                          for t in ("users", "documents", "chunks")}
                report("PASS", "Tables exist", f"rows: {counts}")
                if counts["users"] == 0:
                    report("WARN", "No users yet", "python -m app.auth student@mru.edu pass123 student CSE")
                if counts["documents"] == 0:
                    report("WARN", "No documents ingested yet", "Add PDFs + manifest.csv, then python -m app.ingest ../data/manifest.csv")

            dim = conn.execute(
                "SELECT atttypmod FROM pg_attribute WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
            ).fetchone() if "chunks" in tables else None
            if dim and dim[0] != s.embed_dim:
                report("FAIL", f"chunks.embedding is vector({dim[0]}) but EMBED_DIM={s.embed_dim}",
                       "Make schema.sql and EMBED_DIM match the embedding model, then recreate the table and re-ingest.")
    except Exception as exc:
        report("FAIL", "Postgres connection failed", f"{type(exc).__name__}: {exc}\n"
               "       Is the container running (docker compose ps)? Does DATABASE_URL match the port in docker-compose.yml?")


def check_embeddings() -> None:
    s = get_settings()
    try:
        from .models import embed_query

        v = embed_query("test question about attendance")
        if len(v) == s.embed_dim:
            report("PASS", f"Embeddings work ({s.embed_backend}, dim {len(v)})")
        else:
            report("FAIL", f"Embedding dimension is {len(v)}, expected {s.embed_dim}",
                   "Update EMBED_DIM and the vector(...) size in schema.sql to match.")
    except Exception as exc:
        report("FAIL", "Embedding call failed", f"{type(exc).__name__}: {exc}\n"
               "       Check HF_TOKEN permissions ('Make calls to Inference Providers'), or set EMBED_BACKEND=local.")


def check_reranker() -> None:
    try:
        from .models import rerank_scores

        scores = rerank_scores("minimum attendance for exams", ["Students need 75% attendance.", "The cafeteria opens at 8am."])
        ok = scores[0] > scores[1]
        report("PASS" if ok else "WARN", "Reranker loads and ranks",
               f"scores: {[round(x, 2) for x in scores]}" + ("" if ok else " (relevant passage did not rank first)"))
    except Exception as exc:
        report("FAIL", "Reranker failed", f"{type(exc).__name__}: {exc}")


def check_llm() -> None:
    s = get_settings()
    is_gemini = "generativelanguage.googleapis.com" in s.llm_base_url
    try:
        from .llm import chat

        text, _, _ = chat([{"role": "user", "content": "Reply with the single word: ok"}], max_tokens=300)
        where = s.llm_base_url or "Hugging Face"
        if text.strip():
            report("PASS", f"LLM responds ({s.llm_model} via {where})", f"reply: {text.strip()[:60]!r}")
        else:
            report("WARN", f"LLM reachable but returned empty text ({s.llm_model})",
                   "Reasoning models can use the whole token budget thinking. Set LLM_REASONING_EFFORT=low to .env, "
                   "or raise LLM_MAX_TOKENS.")
    except Exception as exc:
        text = str(exc)
        lowered = text.lower()
        if not s.llm_api_key and s.llm_base_url:
            hint = ("LLM_API_KEY is empty. Paste your key into .env"
                    + (" (free key: aistudio.google.com/apikey)." if is_gemini
                       else " (any OpenAI-compatible provider)."))
        elif any(k in lowered for k in ("api key", "api_key", "401", "403", "unauthenticated", "permission_denied")):
            hint = ("The key was rejected or lacks access to this model. Check LLM_API_KEY in .env"
                    + (" and confirm LLM_MODEL is still listed at ai.google.dev/gemini-api/docs/models."
                       if is_gemini else "."))
        elif any(code in text for code in ("503", "429", "UNAVAILABLE", "overloaded", "rate limit")):
            hint = ("The provider is rate limiting or overloaded (503/429), not a configuration error. "
                    "Retry shortly, set LLM_FALLBACK_MODEL, or switch LLM_MODEL in .env.")
        elif any(k in lowered for k in ("404", "not found", "does not exist", "invalid model")):
            hint = ("LLM_MODEL was not found at this endpoint. Gemini retires model names: pick a current one "
                    "from ai.google.dev/gemini-api/docs/models and set LLM_MODEL in .env." if is_gemini
                    else "LLM_MODEL was not found at this endpoint; pick a current model name in .env.")
        else:
            hint = ("Set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL in .env (Gemini, Ollama or Groq; see README). "
                    "Hugging Face chat models need purchased credits.")
        report("FAIL", "LLM call failed", f"{type(exc).__name__}: {text[:300]}\n       {hint}")


def check_docling() -> None:
    try:
        from docling.chunking import HybridChunker  # noqa: F401
        from docling.document_converter import DocumentConverter  # noqa: F401

        report("PASS", "Docling imports")
    except Exception as exc:
        report("FAIL", "Docling import failed", f"{type(exc).__name__}: {exc}")


def main() -> None:
    for step in (check_python, check_env, check_database, check_docling, check_embeddings, check_reranker, check_llm):
        step()
    fails = sum(1 for s, _ in results if s == "FAIL")
    warns = sum(1 for s, _ in results if s == "WARN")
    print(f"\n{len(results) - fails - warns} passed, {warns} warnings, {fails} failed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
