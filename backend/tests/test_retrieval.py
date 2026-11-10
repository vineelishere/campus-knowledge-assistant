from app.retrieval import _fts_query


def test_drops_stopwords_and_short_tokens():
    q = _fts_query("What is the minimum attendance required for exams?")
    assert q is not None
    tokens = q.split(" | ")
    assert "attendance" in tokens
    assert "minimum" in tokens
    assert "exams" in tokens
    for stop in ("what", "the", "for", "is"):
        assert stop not in tokens


def test_keeps_course_codes_case_insensitively():
    # Course codes are the main reason this uses the 'simple' tsvector config.
    assert _fts_query("prerequisites for CS301?") == "prerequisites | cs301"


def test_deduplicates_tokens():
    assert _fts_query("attendance attendance attendance") == "attendance"


def test_returns_none_when_there_is_nothing_meaningful():
    assert _fts_query("is it on?") is None
    assert _fts_query("") is None


def test_output_is_safe_for_to_tsquery():
    # Only \w tokens are emitted, so quotes/operators from user input cannot reach to_tsquery.
    q = _fts_query("attendance & | ! ( ) ' ; DROP TABLE chunks;")
    assert q is not None
    for token in q.split(" | "):
        assert token.isalnum()
