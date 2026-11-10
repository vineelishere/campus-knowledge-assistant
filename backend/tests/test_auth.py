from types import SimpleNamespace

import jwt
import pytest

import app.auth as auth


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setattr(
        auth, "get_settings", lambda: SimpleNamespace(jwt_secret="test-secret-0123456789abcdef0123456789abcdef", jwt_expire_minutes=60)
    )


def test_role_rank_matches_document_access_levels():
    # access_level <= rank is the permission rule, so the ordering itself is load-bearing.
    assert auth.ROLE_RANK == {"student": 0, "professor": 1, "admin": 2}
    assert auth.ROLE_RANK["student"] < auth.ROLE_RANK["professor"] < auth.ROLE_RANK["admin"]


def test_password_hash_is_salted_and_verifies():
    hashed = auth.hash_password("pass123")
    assert hashed != "pass123"
    assert auth.verify_password("pass123", hashed)
    assert not auth.verify_password("wrong", hashed)
    assert auth.hash_password("pass123") != hashed


def test_token_carries_subject_and_role():
    token = auth.create_token({"id": 42, "role": "professor"})
    payload = jwt.decode(token, "test-secret-0123456789abcdef0123456789abcdef", algorithms=["HS256"])
    assert payload["sub"] == "42"
    assert payload["role"] == "professor"
    assert "exp" in payload


def test_create_user_rejects_unknown_role_before_touching_the_database():
    with pytest.raises(ValueError):
        auth.create_user("x@example.com", "pw", "dean")
