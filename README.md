# Campus Knowledge Assistant

RAG over university policies, syllabi and handbooks. Answers cite the exact document and page, respect
role-based access, refuse to answer from outdated policies, and say "I do not know" when the documents
don't support an answer.

```
PDF -> Docling -> chunks -> embeddings -> Postgres (pgvector + FTS)
Question -> JWT role -> permission filter -> vector + FTS -> RRF -> cross-encoder rerank
         -> freshness check (policy_key versions) -> Hugging Face LLM (JSON, cited passage numbers)
         -> citation validation -> answer | "I do not know"   (every step traced + logged)
```

## Layout
| Path | What |
|---|---|
| `backend/app/schema.sql` | tables: users, documents, chunks (pgvector + tsvector), query_log |
| `backend/app/ingest.py` | Docling extraction + HybridChunker, embeddings, manifest-driven metadata |
| `backend/app/retrieval.py` | permission-filtered hybrid search + RRF + rerank |
| `backend/app/freshness.py` | outdated-version detection (shared `policy_key`, newest `year` wins) |
| `backend/app/llm.py`, `service.py` | grounded answering, citation validation, abstention, logging |
| `backend/app/main.py` | FastAPI: `/auth/login`, `/ask`, `/me`, `/admin/metrics` |
| `eval/` | question set + `run_eval.py` (own metrics + Ragas) |
| `frontend/` | Next.js student UI |

## Run locally
```bash
cp .env.example .env                   # set JWT_SECRET; HF_TOKEN only for HF backends; optional Langfuse keys
docker compose up -d db                # Postgres + pgvector, schema auto-applied
cd backend && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put PDFs in data/pdfs/ and describe them in data/manifest.csv
python -m app.ingest ../data/manifest.csv
python -m app.auth student@mru.edu pass123 student CSE
python -m app.auth prof@mru.edu pass123 professor CSE
python -m app.auth admin@mru.edu pass123 admin
uvicorn app.main:app --reload          # http://localhost:8000/docs
cd ../frontend && cp .env.local.example .env.local && npm install && npm run dev
```

## How the key requirements are met
- **Permission filter before search:** `access_level <= role_rank` is in the WHERE clause of both the
  vector and full-text queries, so restricted chunks never reach the reranker or the LLM. Role is
  re-read from the DB on every request.
- **Outdated documents:** documents with the same `policy_key` are versions, ordered by `year` and then
  `ingested_at` (so a re-ingested copy of a policy also supersedes the earlier one). Older versions are
  dropped from the evidence; the newest version is searched directly. If a superseded version is the
  *strongest* evidence for the question (it outranks every current passage) and the newest version has no
  relevant passage, the system refuses (`stale_only`) and tells the student a newer version exists. It
  never answers an outdated-policy question from weaker, unrelated current chunks.
- **No hallucination:** abstains when (a) no candidates, (b) best rerank score < `MIN_RERANK_SCORE`,
  (c) the LLM says unanswerable, or (d) it returns no valid citation numbers.
- **Observability:** `@observe` spans for search, rerank, freshness and generation go to Langfuse; the
  same outcome data is in `query_log`, exposed at `GET /admin/metrics` (failed searches, unsupported
  answers, errors, p50/p95 latency, cost). Statuses are `answered`, `abstained`, `stale_only`,
  `unsupported` and `error` — a crash is reported as `error`, never as an abstention, so it cannot be
  mistaken for a correct "I do not know". Build the dashboard in Langfuse, or chart `/admin/metrics`.
- **Abuse resistance:** `/auth/login` throttles failed attempts per email+client (`LOGIN_MAX_ATTEMPTS`
  over `LOGIN_WINDOW_SECONDS`) and returns 429. Document text is wrapped in `<passages>` with an explicit
  instruction that it is data, not instructions, to blunt prompt injection from document content.

## Before you trust it
1. **Metadata is manual by design.** `year`, `policy_key` and `access_level` come from `manifest.csv`.
   Wrong values mean wrong staleness decisions or leaks.
2. **Calibrate `MIN_RERANK_SCORE`** on your eval set: raise it until abstain-correct is high without
   losing answerable questions. On the bundled 50-question set, `0.0` keeps every answerable question and
   blocks all nine unanswerable ones; `-4.0` also separates perfectly (41/41 answerable kept, 0/9
   unanswerable reaching the LLM) but the margin (best unanswerable -4.42 vs worst answerable -3.82) is
   narrow and synthetic. Re-run the sweep on your own documents before changing the default.
3. **Scanned PDFs** need OCR enabled in Docling; ingestion reports "no text extracted" for them.
4. **Verify library versions.** Docling's chunker, Langfuse v3 `observe`, and Ragas metrics change
   between releases. Docling/Ragas and the full pipeline have not been executed here against real PDFs;
   only the pure logic is covered by the unit suite, and `python -m app.check` validates the installed
   stack on your machine.
5. **Evaluation set:** `eval/questions.jsonl` has 5 template rows. Build 50+ from your real PDFs:
   ~30 answerable (course prerequisites, attendance, grading, fees), ~8 version-conflict (2024 vs 2026),
   ~6 should-abstain, ~6 RBAC (same question as student vs professor). Write `reference` answers from
   the documents yourself. Run `python ../eval/run_eval.py ../eval/questions.jsonl --ragas`; it exits
   non-zero on any access-control or stale-document leak, so use it in CI.

## Deploy
- **Database:** Neon, Supabase or RDS Postgres with `pgvector`; run `schema.sql`.
- **API:** build `backend/Dockerfile` and run on Render, Railway or Fly.io. Set `.env` values as secrets,
  `ALLOWED_ORIGINS` to the frontend URL. Ingest from a one-off job or admin shell.
  The embedding and reranker models run on CPU in-process; give the container 2-4 GB RAM.
- **Frontend:** import `frontend/` into Vercel, set `NEXT_PUBLIC_API_URL` to the API URL.

## Free Hugging Face setup
- Create a free token at huggingface.co/settings/tokens (enable "Make calls to Inference Providers") and put it in `HF_TOKEN`.
- The answer LLM (`LLM_MODEL`) and embeddings (`EMBED_BACKEND=hf_api`) run on the hosted free tier; the reranker is a small local model.
- Free-tier models and limits change. If a call fails with a model/provider error, switch `LLM_MODEL` to another instruct model
  listed on huggingface.co/models?inference_provider=all&pipeline_tag=text-generation, or set `HF_PROVIDER=hf-inference`.
- Ingestion makes one embedding call per chunk, so large PDF sets can hit rate limits; the code retries with backoff, or
  set `EMBED_BACKEND=local` to embed on your machine instead. Use the same backend for ingestion and querying.
- Changing `EMBED_MODEL` to one with a different dimension requires changing `vector(384)` in `schema.sql` and re-ingesting.
- Small open models follow the "answer only from passages" rule less reliably than larger ones, so run the eval set and tune `MIN_RERANK_SCORE`.

## Sample data and self-check
- `data/pdfs/` ships with six **fictional** sample PDFs (stamped "SAMPLE - FICTIONAL TEST DATA" on every page) so you can test
  without collecting real documents: Academic Regulations 2024 and 2026 (conflicting rules, for outdated-document detection),
  a CSE syllabus, hostel/campus rules, a faculty-only handbook (access level 1) and an admin-only exam-cell document (level 2).
  `data/manifest.csv` already describes them. `eval/questions.jsonl` holds 50 questions with reference answers written against them
  (30 answerable, 8 version-conflict, 6 should-abstain, 6 access-control).
- To use real documents, replace the PDFs and the manifest rows, and rewrite the questions and references from the real text.
- After setup, run `python -m app.check` (from `backend/`). It tests Python version, `.env`, the database connection, pgvector, tables,
  embeddings, the reranker, the LLM call and Docling, and prints a hint for each failure.
- Suggested order: `python -m app.check` -> `python -m app.ingest ../data/manifest.csv` -> create users -> `python ../eval/run_eval.py ../eval/questions.jsonl`.

## Answer LLM options (updated)
The shipped config uses **Google Gemini** for answer generation. Retrieval never calls an LLM: embeddings
(`BAAI/bge-small-en-v1.5`) and the reranker (`ms-marco-MiniLM-L-6-v2`) run locally on CPU, so only the final
grounded answer costs an API call.
- **Gemini (default):** free key at aistudio.google.com/apikey, then in `.env`
  `LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`, `LLM_API_KEY=<key>`,
  `LLM_MODEL=gemini-3.8-flash`, `LLM_REASONING_EFFORT=low`. This is Gemini's OpenAI-compatible endpoint,
  so no code changes are involved. Model names are retired often — if a call fails with "model not found",
  pick a current one from ai.google.dev/gemini-api/docs/models and set `LLM_MODEL`.
  Gemini 3.x cannot switch thinking off; `LLM_REASONING_EFFORT=low` keeps the JSON answer inside
  `LLM_MAX_TOKENS`. Gemini 2.5 models accept `none` to disable thinking entirely.
- **Ollama (local, free):** install from ollama.com, `ollama pull qwen2.5:7b`, then
  `LLM_BASE_URL=http://localhost:11434/v1`, `LLM_API_KEY=ollama`, `LLM_MODEL=qwen2.5:7b`. Needs about 8 GB RAM; CPU answers take 10-40 s.
- **Groq (free API key):** key from console.groq.com, `LLM_BASE_URL=https://api.groq.com/openai/v1`,
  `LLM_MODEL=llama-3.3-70b-versatile`. Free-tier limits and model names change; check Groq's console.
- Any other OpenAI-compatible endpoint works the same way. Leave `LLM_BASE_URL` empty to use Hugging Face instead.
- Keeping embeddings local is deliberate: switching to Gemini embeddings (`gemini-embedding-001`, 768+ dims)
  would mean changing `vector(384)` in `schema.sql` and re-ingesting every document. Run
  `python -m app.check` after changing any of these — it now names the provider that is misconfigured.

## Testing as a user
0. **Unit tests** (no database, no LLM, no network; CI runs them on every push):
   `cd backend && pip install -r requirements-test.txt && python -m pytest`. They cover the FTS query
   builder, JSON/citation validation, version supersession and `stale_only`, pipeline statuses,
   role ranks and password hashing. `python -m compileall -q backend/app eval` checks the rest.
1. API smoke test (3 roles, 10 checks): `python eval/smoke_api.py` (API running, sample data ingested, users created).
   It needs a working answer LLM: if `/ask` returns `status: "error"`, the provider is down or throttled —
   confirm with `python -m app.check` before blaming the pipeline.
2. In the browser (http://localhost:3000): log in as student, professor, admin (password `pass123`) and ask:
   - everyone: "Can I take the machine learning course before taking statistics?" -> answer + source citing the CSE syllabus
   - everyone: "What is the minimum attendance required to sit for end semester exams?" -> 80%, cites Academic Regulations 2026 only
   - student: "What are the promotion criteria for associate professors?" -> "I do not know"; professor: a real answer citing the Faculty Handbook
   - professor: "Who can award grace marks and how many?" -> "I do not know"; admin: answer citing Exam Cell Procedures 2026
   - admin: a "dashboard" link appears next to "log out"; student and professor do not see it.

## Tracing and dashboard
- Langfuse: create a project at cloud.langfuse.com (EU) or us.cloud.langfuse.com (US), copy the public and secret keys into
  `.env` (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` matching your region) and remove `LANGFUSE_TRACING_ENABLED=false`.
  Each question becomes a trace with spans for search, rerank, freshness check and generation (model + token usage), tagged
  `answered`, `abstained`, `unsupported`, `stale_only` or `error`, with `reason`, `top_score`, `latency_ms` in the metadata.
- Built-in dashboard: log in as admin and open `/admin` in the frontend. It reads `GET /admin/metrics` (data from the `query_log` table):
  question counts, % answered, failed searches, unsupported answers, median and p95 response time, cost, per-day table, list of unanswered questions.
- Cost is $0 on free tiers. To see a cost figure, set `INPUT_PRICE_PER_MTOK` / `OUTPUT_PRICE_PER_MTOK` to the model's paid-tier prices.

## Ragas
`pip install -r backend/requirements-eval.txt`, then from `backend/`: run a full eval WITHOUT `--reuse` (so full passages are stored),
then `python ../eval/run_ragas.py eval_results.jsonl --n 12`. Uses the LLM from `.env` as judge and the local embedding model.

## Deployment (updated)
Free-tier facts change; confirm on each provider's pricing page.
- Database: Neon (free: 1 GB, suspends after 5 min idle) or Supabase. Create the project, run `CREATE EXTENSION vector;` and apply
  `backend/app/schema.sql` (`psql "<connection string>" -f backend/app/schema.sql`). Create users and ingest from your own machine by setting
  `DATABASE_URL` to the cloud database (ingestion needs Docling and is too heavy to run on a small host).
- API: `backend/Dockerfile` is a slim image (CPU torch, the two local models baked in, listens on `$PORT`). It needs roughly 1 GB RAM.
  Hugging Face Docker Spaces require a paid plan. Render's free web service sleeps after 15 minutes and may not have enough memory.
  Cheap options: a small VPS (2 GB RAM), or run the API on your own machine and expose it with a Cloudflare quick tunnel.
- Frontend: import `frontend/` into Vercel, set `NEXT_PUBLIC_API_URL` to the API URL; set `ALLOWED_ORIGINS` on the API to the Vercel URL.
- Required API environment variables: `DATABASE_URL`, `JWT_SECRET`, `ENVIRONMENT=production`, `LLM_BASE_URL`,
  `LLM_API_KEY`, `LLM_MODEL`, `EMBED_BACKEND=local`, `RERANK_BACKEND=local`, `MIN_RERANK_SCORE`,
  `ALLOWED_ORIGINS`, and the `LANGFUSE_*` keys. With `ENVIRONMENT=production` the API refuses to start
  while `JWT_SECRET` is still a default value.
