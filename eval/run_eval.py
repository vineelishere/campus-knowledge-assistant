"""Evaluate retrieval, citations, abstention and access control over eval/questions.jsonl.

Run from the backend/ directory (so `app` imports and .env resolve):
    python ../eval/run_eval.py ../eval/questions.jsonl --ragas

Own metrics (no extra LLM calls, deterministic):
  retrieval_hit      an expected doc appears in the reranked top-N
  citation_correct   every cited doc is in expected_docs and none is in forbidden_docs
  abstain_correct    answerable <-> the system answered (so unanswerable questions must abstain)
  rbac_leak          a forbidden doc was retrieved or cited (must be 0)
  stale_leak         an older version listed in forbidden_docs was cited (must be 0)
--ragas additionally runs Ragas context_precision / context_recall / faithfulness on answerable
questions that have a reference answer. Ragas' API changes between releases; check the
version in requirements.txt if the import below fails.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from app.auth import ROLE_RANK  # noqa: E402
from app.db import connection  # noqa: E402
from app.service import ask  # noqa: E402


def fake_user(role: str) -> dict:
    return {"id": 0, "role": role, "rank": ROLE_RANK[role], "email": f"eval-{role}"}


def contexts_for(citations: list[dict]) -> list[str]:
    return [c["excerpt"] for c in citations]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("--ragas", action="store_true")
    ap.add_argument("--out", default="eval_results.jsonl")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait between questions (free-tier rate limits)")
    ap.add_argument("--reuse", help="previous results file: questions it answered are kept as-is, so only the rest call the LLM")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.questions).read_text().splitlines() if l.strip()]
    if len(rows) < 50:
        print(f"WARNING: only {len(rows)} questions; the target is at least 50.\n")

    prev: dict = {}
    if args.reuse and Path(args.reuse).exists():
        prev = {r["id"]: r for r in (json.loads(l) for l in Path(args.reuse).read_text().splitlines() if l.strip())}

    results, by_cat = [], defaultdict(lambda: defaultdict(list))
    called, reused = False, 0
    for q in rows:
        p = prev.get(q["id"])
        if p and p["status"] == "answered" and p["question"] == q["question"]:
            reused += 1
            status, reason, top_score, answer = "answered", None, p.get("top_score"), p["answer"]
            cited, retrieved, contexts = set(p["cited"]), set(p["retrieved"]), p["contexts"]
        else:
            if called and args.delay:
                time.sleep(args.delay)
            a = ask(fake_user(q["role"]), q["question"])
            called = True
            status, reason, top_score, answer = a.status, a.reason, a.top_score, a.answer
            cited = {c["title"] for c in a.citations}
            retrieved = {r["title"] for r in a.retrieved}
            contexts = a.contexts
        expected = set(q.get("expected_docs", []))
        forbidden = set(q.get("forbidden_docs", []))
        answered = status == "answered"
        errored = status == "error"  # a crash is never a correct abstention

        m = {
            "retrieval_hit": (bool(expected & retrieved) if expected else None),
            "citation_correct": ((answered and cited <= expected and not (cited & forbidden)) if q["expected_answerable"] else None),
            "abstain_correct": (answered == q["expected_answerable"]) and not errored,
            "rbac_leak": bool(forbidden & (retrieved | cited)) if q["category"] == "rbac" else None,
            "stale_leak": bool(forbidden & cited) if q["category"] == "version_conflict" else None,
        }
        for k, v in m.items():
            if v is not None:
                by_cat[q["category"]][k].append(v)
                by_cat["ALL"][k].append(v)
        results.append({**q, "status": status, "reason": reason, "top_score": top_score, "answer": answer, "cited": sorted(cited),
                        "retrieved": sorted(retrieved), "metrics": m, "contexts": contexts})
        print(f"{q['id']:>5} {q['category']:<17} {status:<11} {m}")

    Path(args.out).write_text("\n".join(json.dumps(r) for r in results))
    if reused:
        print(f"\n(reused {reused} previously answered results; only the other {len(rows) - reused} were re-asked)")

    print("\n=== Summary (rate of True) ===")
    for cat, ms in by_cat.items():
        print(cat, {k: f"{sum(v)}/{len(v)}" for k, v in ms.items()})
    leaks = sum(by_cat["ALL"].get("rbac_leak", [])) + sum(by_cat["ALL"].get("stale_leak", []))
    errors = sum(1 for r in results if r["status"] == "error")
    print(f"\nLeaks (must be 0): {leaks}   Errors (must be 0): {errors}")
    threshold_sweep(results)
    reasons = defaultdict(int)
    for r in results:
        if r["status"] != "answered":
            reasons[f'{r["status"]}/{r["reason"]}'] += 1
    print("Not-answered breakdown:", dict(reasons))

    if args.ragas:
        run_ragas(results)

    try:  # send pending Langfuse traces before this short-lived process exits
        from langfuse import get_client

        get_client().flush()
    except Exception:
        pass
    sys.exit(1 if (leaks or errors) else 0)


def threshold_sweep(results: list[dict]) -> None:
    """For each candidate MIN_RERANK_SCORE: how many answerable questions would pass the cutoff
    vs. how many unanswerable ones would still reach the LLM (which must then refuse)."""
    ans = [r["top_score"] for r in results if r["expected_answerable"] and r.get("top_score") is not None]
    unans = [r["top_score"] for r in results if not r["expected_answerable"] and r.get("top_score") is not None]
    if not ans:
        return
    print("\n=== MIN_RERANK_SCORE sweep (top reranker score per question) ===")
    print(f"answerable top scores: min {min(ans):.1f}, median {sorted(ans)[len(ans) // 2]:.1f}")
    if unans:
        print(f"unanswerable top scores: max {max(unans):.1f}, median {sorted(unans)[len(unans) // 2]:.1f}")
    print(f"{'cutoff':>7} {'answerable kept':>16} {'unanswerable reaching LLM':>26}")
    for t in (-8, -6, -5, -4, -3, -2, -1, 0, 1, 2):
        print(f"{t:>7} {sum(s >= t for s in ans):>9}/{len(ans):<6} {sum(s >= t for s in unans):>19}/{len(unans)}")


def run_ragas(results: list[dict]) -> None:
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import context_precision, context_recall, faithfulness

    data = [
        {
            "user_input": r["question"],
            "retrieved_contexts": r["contexts"],
            "response": r["answer"],
            "reference": r["reference"],
        }
        for r in results
        if r["expected_answerable"] and r["status"] == "answered" and r["reference"] and not r["reference"].startswith("FILL IN")
    ]
    if not data:
        print("\nRagas: no answered questions with reference answers; fill in `reference` first.")
        return
    report = evaluate(EvaluationDataset.from_list(data), metrics=[context_precision, context_recall, faithfulness])
    print("\n=== Ragas ===")
    print(report)


if __name__ == "__main__":
    main()
