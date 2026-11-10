from app.llm import _coerce_citations, _format_passages, _parse
from app.retrieval import Hit


def hit(title: str, year: int, page: int | None = 1, text: str = "body") -> Hit:
    return Hit(
        chunk_id=1,
        document_id=1,
        title=title,
        year=year,
        policy_key=None,
        department=None,
        page=page,
        text=text,
    )


def test_parse_plain_json():
    assert _parse('{"answerable": true, "answer": "yes", "citations": [1]}') == {
        "answerable": True,
        "answer": "yes",
        "citations": [1],
    }


def test_parse_strips_reasoning_block():
    raw = ' thinkingthe answer is in passage 1</think>{"answerable": true, "answer": "ok", "citations": [1]}'
    assert _parse(raw)["answer"] == "ok"


def test_parse_tolerates_code_fences_and_prose():
    raw = 'Here you go:\n```json\n{"answerable": false, "answer": "", "citations": []}\n```'
    assert _parse(raw)["answerable"] is False


def test_parse_returns_empty_dict_for_garbage():
    assert _parse("I could not answer that.") == {}
    assert _parse("{not json}") == {}


def test_citations_accept_common_shapes():
    assert _coerce_citations([1, "2", "P3", "[Passage 2]"], 3) == [1, 2, 3]
    assert _coerce_citations("2", 3) == [2]


def test_citations_reject_out_of_range_and_non_numbers():
    # Out-of-range numbers are the dangerous case: they would index a passage that was never sent.
    assert _coerce_citations([0, 4, 99, -1], 3) == []
    assert _coerce_citations([True, False], 3) == []
    assert _coerce_citations(None, 3) == []
    assert _coerce_citations({"n": 1}, 3) == []


def test_citations_deduplicate():
    assert _coerce_citations([1, 1, "P1"], 3) == [1]


def test_format_passages_numbers_from_one():
    text = _format_passages([hit("Regs", 2026, 3), hit("Syllabus", 2025, None)])
    assert "[Passage 1] Regs (2026, page 3)" in text
    assert "[Passage 2] Syllabus (2025)" in text
