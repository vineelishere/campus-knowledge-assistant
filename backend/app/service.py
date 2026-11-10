"""The full question-answering pipeline: permission-filtered retrieval -> rerank ->
freshness check -> grounded LLM answer -> citation validation -> logging."""
import sys
import time
from dataclasses import dataclass, field

from .config import get_settings
from .db import connection
from .freshness import resolve_freshness
from .llm import generate
from .retrieval import Hit, hybrid_search, rerank
from .tracing import observe, update_trace

IDK = "I do not know. I could not find this in the documents available to you."


@dataclass
class Answer:
    answer: str
    status: str  # answered | abstained | stale_only | unsupported | error
    citations: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reason: str | None = None
    top_score: float | None = None
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    document_ids: list[int] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)  # for evaluation
    contexts: list[str] = field(default_factory=list)  # full passages sent to the LLM (for Ragas)


def _cite(h: Hit, n: int) -> dict:
    return {
        "n": n,
        "document_id": h.document_id,
        "title": h.title,
        "year": h.year,
        "page": h.page,
        "department": h.department,
        "source_url": h.source_url,
        "excerpt": h.text[:400],
    }


def _log(user: dict, question: str, a: Answer) -> None:
    try:
        with connection() as conn:
            conn.execute(
                """INSERT INTO query_log (user_id, role, question, status, reason, top_score, latency_ms,
                                          input_tokens, output_tokens, cost_usd, document_ids)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (user["id"], user["role"], question, a.status, a.reason, a.top_score, a.latency_ms,
                 a.input_tokens, a.output_tokens, a.cost_usd, a.document_ids),
            )
            conn.commit()
    except Exception as exc:
        # Logging must never fail a request, but it must not be invisible either.
        print(f"warning: could not write query_log: {type(exc).__name__}: {exc}", file=sys.stderr)


@observe(name="ask")
def ask(user: dict, question: str) -> Answer:
    s = get_settings()
    t0 = time.perf_counter()
    update_trace(user_id=str(user["id"]), metadata={"role": user["role"]}, input=question)

    ans = Answer(answer=IDK, status="abstained")
    try:
        candidates = hybrid_search(question, user["rank"])
        if not candidates:
            ans.reason = "no_candidates"
        else:
            ranked = rerank(question, candidates)
            ans.retrieved = [{"document_id": h.document_id, "title": h.title, "page": h.page,
                              "year": h.year, "score": h.score} for h in ranked]
            fresh = resolve_freshness(question, ranked, user["rank"])
            ans.warnings = fresh.warnings
            hits = [h for h in fresh.hits if h.score >= s.min_rerank_score][: s.rerank_top_n]
            ans.top_score = hits[0].score if hits else (ranked[0].score if ranked else None)
            ans.contexts = [h.text for h in hits]

            if fresh.stale_only:
                ans.status, ans.reason = "stale_only", "only_outdated_documents"
                ans.answer = ("I found only an outdated version of the relevant policy, so I will not answer "
                              "from it. Please check with the academic office for the current rules.")
            elif not hits:
                ans.reason = "low_relevance"
            else:
                out = generate(question, hits)
                ans.input_tokens, ans.output_tokens = out.input_tokens, out.output_tokens
                ans.cost_usd = (out.input_tokens * s.input_price_per_mtok
                                + out.output_tokens * s.output_price_per_mtok) / 1_000_000
                if out.answerable:
                    ans.status, ans.answer = "answered", out.answer
                    ans.citations = [_cite(hits[n - 1], n) for n in out.cited]
                    ans.document_ids = sorted({hits[n - 1].document_id for n in out.cited})
                else:
                    ans.status, ans.reason = "unsupported", "llm_declined_or_uncited"
    except Exception as exc:  # surface a safe message, but keep the failure distinct from abstention
        ans.status, ans.reason = "error", f"{type(exc).__name__}: {exc}"[:300]
        ans.answer = "Something went wrong while searching. Please try again."
        print(f"error: /ask failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    ans.latency_ms = int((time.perf_counter() - t0) * 1000)
    update_trace(output=ans.answer, metadata={
        "role": user["role"], "status": ans.status, "reason": ans.reason,
        "top_score": ans.top_score, "cost_usd": ans.cost_usd, "latency_ms": ans.latency_ms,
        "stale_warnings": ans.warnings,
    }, tags=[ans.status])
    _log(user, question, ans)
    return ans
