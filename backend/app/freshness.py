"""Outdated-document detection.

Documents that share a policy_key are versions of the same policy (e.g. the 2024 and 2026
prerequisite policy). Only the newest version *visible to this user* may support an answer.

Two rules matter:
* A hit is superseded when its document is older than the newest visible version of the same
  policy_key. Documents with the same year are ordered by ingested_at, so a re-ingested or
  corrected copy of a policy also supersedes the earlier one.
* If the only strong evidence for a question is a superseded version, the system refuses
  (``stale_only``) instead of answering from weaker, unrelated current passages. "Stronger"
  is measured with the reranker score, so a stale hit that outranks every current hit means
  the question was really about the outdated policy.
"""
from dataclasses import dataclass, field

from .config import get_settings
from .db import connection
from .retrieval import Hit, hybrid_search, rerank
from .tracing import observe

# A superseded policy triggers at most this many extra searches per question, so a query that
# touches several outdated policies cannot multiply CPU reranking inside one request.
MAX_STALE_LOOKUPS = 3


@dataclass
class FreshnessResult:
    hits: list[Hit]
    warnings: list[str] = field(default_factory=list)
    stale_only: bool = False  # only an outdated version matched and no current evidence was found


def _latest_versions(policy_keys: list[str], rank: int) -> dict[str, dict]:
    with connection() as conn:
        rows = conn.execute(
            """SELECT DISTINCT ON (policy_key) policy_key, id, year, title
               FROM documents
               WHERE policy_key = ANY(%s) AND access_level <= %s
               ORDER BY policy_key, year DESC, ingested_at DESC""",
            (policy_keys, rank),
        ).fetchall()
    return {r["policy_key"]: r for r in rows}


def _is_superseded(hit: Hit, latest: dict | None) -> bool:
    """True when this hit belongs to an older version of the hit's own policy_key."""
    if not hit.policy_key or latest is None:
        return False
    if hit.year < latest["year"]:
        return True
    # Same year, different document: the newer ingestion wins.
    return hit.year == latest["year"] and hit.document_id != latest["id"]


@observe(name="freshness_check")
def resolve_freshness(question: str, hits: list[Hit], rank: int) -> FreshnessResult:
    keys = sorted({h.policy_key for h in hits if h.policy_key})
    if not keys:
        return FreshnessResult(hits=hits)

    latest = _latest_versions(keys, rank)
    keep: list[Hit] = []
    warnings: list[str] = []
    stale_best: dict[str, tuple[float, str]] = {}  # policy_key -> (best score, title of the stale copy)

    for h in hits:
        latest_entry = latest.get(h.policy_key) if h.policy_key else None
        if _is_superseded(h, latest_entry):
            current = stale_best.get(h.policy_key)
            if current is None or h.score > current[0]:
                stale_best[h.policy_key] = (h.score, h.title)
        else:
            keep.append(h)

    min_score = get_settings().min_rerank_score
    unresolved: list[str] = []
    lookups = 0
    for key, (stale_score, stale_title) in stale_best.items():
        cur = latest[key]
        has_current = any(h.policy_key == key and h.document_id == cur["id"] for h in keep)
        if not has_current and lookups < MAX_STALE_LOOKUPS:
            # Look specifically inside the newest version for an answer.
            lookups += 1
            extra = rerank(question, hybrid_search(question, rank, doc_ids=[cur["id"]]))
            extra = [h for h in extra if h.score >= min_score]
            keep.extend(extra)
            has_current = bool(extra)
        if has_current:
            warnings.append(
                f"An older version ('{stale_title}') was ignored; this answer uses the "
                f"{cur['year']} version, '{cur['title']}'."
            )
        else:
            warnings.append(
                f"Only an outdated version ('{stale_title}') matched your question. "
                f"A newer version ('{cur['title']}') exists but I could not find a relevant "
                f"passage in it."
            )
            unresolved.append(key)

    keep.sort(key=lambda h: h.score, reverse=True)

    # Refuse when the question is really about an outdated policy: the best stale hit outranks
    # everything current that survived. Weaker unrelated passages must not answer it instead.
    best_stale = max((stale_best[k][0] for k in unresolved), default=None)
    best_kept = keep[0].score if keep else None
    stale_only = best_stale is not None and (best_kept is None or best_stale > best_kept)

    return FreshnessResult(hits=keep, warnings=warnings, stale_only=stale_only)
