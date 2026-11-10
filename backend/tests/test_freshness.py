from types import SimpleNamespace

import pytest

import app.freshness as fres
from app.freshness import _is_superseded, resolve_freshness
from app.retrieval import Hit


def hit(chunk_id: int, document_id: int, title: str, year: int, key: str | None, score: float) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        document_id=document_id,
        title=title,
        year=year,
        policy_key=key,
        department=None,
        page=1,
        text=f"text {chunk_id}",
        score=score,
    )


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setattr(fres, "get_settings", lambda: SimpleNamespace(min_rerank_score=0.0))


def _patch_latest(monkeypatch, entries: dict[str, dict]) -> None:
    monkeypatch.setattr(fres, "_latest_versions", lambda keys, rank: {k: v for k, v in entries.items() if k in keys})


def _patch_lookup(monkeypatch, extra_hits: list[Hit], calls: list | None = None) -> None:
    def fake_search(question, rank, doc_ids=None):
        if calls is not None:
            calls.append(doc_ids)
        return list(extra_hits)

    monkeypatch.setattr(fres, "hybrid_search", fake_search)
    monkeypatch.setattr(fres, "rerank", lambda question, hits: hits)


LATEST_2026 = {"regs": {"policy_key": "regs", "id": 2, "year": 2026, "title": "Academic Regulations 2026"}}


def test_documents_without_policy_key_are_untouched(monkeypatch):
    _patch_latest(monkeypatch, {})
    hits = [hit(1, 1, "Hostel Rules", 2026, None, 4.0)]
    result = resolve_freshness("q", hits, rank=0)
    assert result.hits == hits
    assert result.warnings == []
    assert result.stale_only is False


def test_older_version_is_dropped_when_newer_version_is_present(monkeypatch):
    _patch_latest(monkeypatch, LATEST_2026)
    old = hit(1, 1, "Academic Regulations 2024", 2024, "regs", 5.0)
    new = hit(2, 2, "Academic Regulations 2026", 2026, "regs", 1.0)
    result = resolve_freshness("attendance rules", [old, new], rank=0)
    assert [h.chunk_id for h in result.hits] == [2]
    assert "Academic Regulations 2024" in result.warnings[0]
    assert "2026 version" in result.warnings[0]
    # The stale hit outranks the current one, but the current version is present, so this is
    # a normal answer from the new policy, not a stale-only refusal.
    assert result.stale_only is False


def test_looks_inside_newer_version_when_only_old_version_matched(monkeypatch):
    _patch_latest(monkeypatch, LATEST_2026)
    _patch_lookup(monkeypatch, [hit(9, 2, "Academic Regulations 2026", 2026, "regs", 3.0)])
    result = resolve_freshness("attendance rules", [hit(1, 1, "Academic Regulations 2024", 2024, "regs", 5.0)], rank=0)
    assert [h.chunk_id for h in result.hits] == [9]
    assert "was ignored" in result.warnings[0]
    assert result.stale_only is False


def test_stale_only_when_newer_version_has_no_relevant_passage(monkeypatch):
    _patch_latest(monkeypatch, LATEST_2026)
    _patch_lookup(monkeypatch, [])  # nothing relevant inside the 2026 document
    result = resolve_freshness("attendance rules", [hit(1, 1, "Academic Regulations 2024", 2024, "regs", 5.0)], rank=0)
    assert result.hits == []
    assert "Only an outdated version" in result.warnings[0]
    assert result.stale_only is True


def test_refuses_when_stale_evidence_outranks_unrelated_current_passage(monkeypatch):
    _patch_latest(monkeypatch, LATEST_2026)
    _patch_lookup(monkeypatch, [])
    stale = hit(1, 1, "Academic Regulations 2024", 2024, "regs", 5.0)
    unrelated = hit(2, 3, "Hostel and Campus Rules 2026", 2026, None, 1.0)
    result = resolve_freshness("attendance rules", [stale, unrelated], rank=0)
    assert result.stale_only is True  # must not answer the old-policy question from hostel rules
    assert [h.chunk_id for h in result.hits] == [2]


def test_answers_from_current_evidence_that_outranks_the_stale_hit(monkeypatch):
    _patch_latest(monkeypatch, LATEST_2026)
    _patch_lookup(monkeypatch, [])
    stale = hit(1, 1, "Academic Regulations 2024", 2024, "regs", 1.0)
    current = hit(2, 3, "B.Tech CSE Syllabus 2026", 2026, None, 6.0)
    result = resolve_freshness("prerequisites", [stale, current], rank=0)
    assert result.stale_only is False
    assert "Only an outdated version" in result.warnings[0]
    assert [h.chunk_id for h in result.hits] == [2]


def test_same_year_duplicate_document_is_superseded(monkeypatch):
    _patch_latest(monkeypatch, LATEST_2026)
    _patch_lookup(monkeypatch, [])
    # Same policy_key and year as the newest document, but an earlier ingestion.
    duplicate = hit(1, 1, "Academic Regulations 2026 (draft)", 2026, "regs", 4.0)
    result = resolve_freshness("attendance rules", [duplicate], rank=0)
    assert result.hits == []
    assert result.stale_only is True


def test_stale_lookups_are_bounded(monkeypatch):
    entries = {
        f"p{i}": {"policy_key": f"p{i}", "id": 100 + i, "year": 2026, "title": f"Policy {i}"} for i in range(5)
    }
    _patch_latest(monkeypatch, entries)
    calls: list = []
    _patch_lookup(monkeypatch, [], calls)
    hits = [hit(i, i, f"Policy {i} 2024", 2024, f"p{i}", 5.0) for i in range(5)]
    resolve_freshness("q", hits, rank=0)
    assert len(calls) == fres.MAX_STALE_LOOKUPS


def test_is_superseded_handles_missing_policy_key():
    assert _is_superseded(hit(1, 1, "X", 2024, None, 0.0), LATEST_2026["regs"]) is False
