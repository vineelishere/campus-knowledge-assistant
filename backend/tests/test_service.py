from types import SimpleNamespace

import pytest

import app.service as svc
from app.freshness import FreshnessResult
from app.retrieval import Hit

SETTINGS = SimpleNamespace(
    min_rerank_score=0.0,
    rerank_top_n=6,
    input_price_per_mtok=0.0,
    output_price_per_mtok=0.0,
)
USER = {"id": 1, "role": "student", "rank": 0, "email": "s@example.com"}


def hit(chunk_id: int, document_id: int, title: str, score: float) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        document_id=document_id,
        title=title,
        year=2026,
        policy_key=None,
        department=None,
        page=7,
        text=f"body {chunk_id}",
        score=score,
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(svc, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr(svc, "_log", lambda *a, **k: None)
    monkeypatch.setattr(svc, "update_trace", lambda **k: None)
    monkeypatch.setattr(svc, "rerank", lambda question, hits, top_n=None: hits)


def _candidates(monkeypatch, hits):
    monkeypatch.setattr(svc, "hybrid_search", lambda question, rank: hits)
    monkeypatch.setattr(svc, "resolve_freshness", lambda q, h, rank: FreshnessResult(hits=h))


def test_no_candidates_abstains(monkeypatch):
    _candidates(monkeypatch, [])
    ans = svc.ask(USER, "anything")
    assert ans.status == "abstained"
    assert ans.reason == "no_candidates"
    assert ans.answer == svc.IDK


def test_low_relevance_abstains(monkeypatch):
    monkeypatch.setattr(svc, "get_settings", lambda: SimpleNamespace(**{**SETTINGS.__dict__, "min_rerank_score": 2.0}))
    _candidates(monkeypatch, [hit(1, 1, "Doc", 1.0)])
    ans = svc.ask(USER, "anything")
    assert ans.status == "abstained"
    assert ans.reason == "low_relevance"
    assert ans.contexts == []


def test_stale_only_short_circuits_before_generation(monkeypatch):
    monkeypatch.setattr(svc, "hybrid_search", lambda question, rank: [hit(1, 1, "Old", 5.0)])
    monkeypatch.setattr(
        svc, "resolve_freshness",
        lambda q, h, rank: FreshnessResult(hits=[], warnings=["outdated"], stale_only=True),
    )
    called = []
    monkeypatch.setattr(svc, "generate", lambda *a: called.append(a))
    ans = svc.ask(USER, "old policy question")
    assert ans.status == "stale_only"
    assert "outdated version" in ans.answer
    assert ans.warnings == ["outdated"]
    assert called == []


def test_llm_declining_is_unsupported(monkeypatch):
    _candidates(monkeypatch, [hit(1, 1, "Doc", 3.0)])
    monkeypatch.setattr(
        svc, "generate",
        lambda q, hits: SimpleNamespace(answerable=False, answer="", cited=[], input_tokens=5, output_tokens=2),
    )
    ans = svc.ask(USER, "anything")
    assert ans.status == "unsupported"
    assert ans.reason == "llm_declined_or_uncited"
    assert ans.citations == []


def test_answer_maps_citation_numbers_to_the_right_passages(monkeypatch):
    _candidates(monkeypatch, [hit(1, 10, "First", 5.0), hit(2, 20, "Second", 4.0)])
    monkeypatch.setattr(
        svc, "generate",
        lambda q, hits: SimpleNamespace(answerable=True, answer="Use the 2026 rule.", cited=[2],
                                        input_tokens=100, output_tokens=20),
    )
    ans = svc.ask(USER, "attendance?")
    assert ans.status == "answered"
    assert ans.answer == "Use the 2026 rule."
    assert ans.citations[0]["n"] == 2
    assert ans.citations[0]["title"] == "Second"
    assert ans.citations[0]["document_id"] == 20
    assert ans.document_ids == [20]
    assert ans.contexts == ["body 1", "body 2"]


def test_crash_is_reported_as_error_not_abstention(monkeypatch):
    def boom(question, rank):
        raise RuntimeError("database down")

    monkeypatch.setattr(svc, "hybrid_search", boom)
    ans = svc.ask(USER, "anything")
    assert ans.status == "error"
    assert ans.reason.startswith("RuntimeError")
    assert "went wrong" in ans.answer
