"""Permission-filtered hybrid retrieval (pgvector + Postgres FTS) fused with RRF, then reranked."""
import re
from dataclasses import dataclass

from .config import get_settings
from .db import connection
from .models import embed_query, rerank_scores
from .tracing import observe

_STOP = {
    "the", "and", "for", "can", "are", "but", "not", "you", "any", "does", "what", "when",
    "how", "who", "why", "this", "that", "with", "from", "have", "has", "will", "should",
    "could", "would", "into", "about", "than", "then", "there", "their", "they", "them",
    "take", "taking", "before", "after", "need", "want", "get", "its", "our", "your",
}
RRF_K = 60


@dataclass
class Hit:
    chunk_id: int
    document_id: int
    title: str
    year: int
    policy_key: str | None
    department: str | None
    page: int | None
    text: str
    source_url: str | None = None
    score: float = 0.0  # reranker logit


def _fts_query(question: str) -> str | None:
    """OR-join the meaningful tokens so a natural-language question still matches exact
    course codes / policy names. Tokens are \\w+ only, so the string is safe for to_tsquery."""
    toks = [t.lower() for t in re.findall(r"\w+", question) if len(t) > 2 and t.lower() not in _STOP]
    return " | ".join(dict.fromkeys(toks)) or None


@observe(name="hybrid_search")
def hybrid_search(question: str, rank: int, doc_ids: list[int] | None = None, k: int | None = None) -> list[Hit]:
    s = get_settings()
    k = k or s.candidates_per_retriever
    qvec = embed_query(question)
    tsq = _fts_query(question)

    # The permission filter (access_level <= rank) is part of every query: documents the
    # user may not see never enter the candidate set, the reranker, or the LLM prompt.
    doc_clause = "AND c.document_id = ANY(%(docs)s)" if doc_ids else ""
    params = {"rank": rank, "k": k, "q": qvec, "docs": doc_ids, "tsq": tsq}

    with connection() as conn:
        vec_ids = [
            r["id"]
            for r in conn.execute(
                f"""SELECT c.id FROM chunks c
                    WHERE c.access_level <= %(rank)s {doc_clause}
                    ORDER BY c.embedding <=> %(q)s LIMIT %(k)s""",
                params,
            ).fetchall()
        ]
        fts_ids: list[int] = []
        if tsq:
            fts_ids = [
                r["id"]
                for r in conn.execute(
                    f"""SELECT c.id FROM chunks c
                        WHERE c.access_level <= %(rank)s {doc_clause}
                          AND c.tsv @@ to_tsquery('simple', %(tsq)s)
                        ORDER BY ts_rank_cd(c.tsv, to_tsquery('simple', %(tsq)s)) DESC
                        LIMIT %(k)s""",
                    params,
                ).fetchall()
            ]

        fused: dict[int, float] = {}
        for ids in (vec_ids, fts_ids):
            for pos, cid in enumerate(ids):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + pos + 1)
        if not fused:
            return []

        rows = conn.execute(
            """SELECT c.id AS chunk_id, c.document_id, c.page, c.text,
                      d.title, d.year, d.policy_key, d.department, d.source_url
               FROM chunks c JOIN documents d ON d.id = c.document_id
               WHERE c.id = ANY(%s)""",
            (list(fused),),
        ).fetchall()

    hits = [Hit(**r) for r in rows]
    hits.sort(key=lambda h: fused[h.chunk_id], reverse=True)
    return hits


@observe(name="rerank")
def rerank(question: str, hits: list[Hit], top_n: int | None = None) -> list[Hit]:
    top_n = top_n or get_settings().rerank_top_n
    scores = rerank_scores(question, [h.text for h in hits])
    for h, sc in zip(hits, scores):
        h.score = sc
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_n]
