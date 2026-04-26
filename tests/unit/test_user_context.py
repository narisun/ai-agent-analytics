"""Unit tests for UserContext value type."""
import dataclasses

import pytest

from src.domain.types import UserContext


class TestUserContext:
    def test_is_frozen(self):
        ctx = UserContext(user_id="u1", tenant_id="t1", auth_token="tok")
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.user_id = "u2"  # type: ignore[misc]

    def test_equality_by_value(self):
        a = UserContext(user_id="u1", tenant_id="t1", auth_token="tok")
        b = UserContext(user_id="u1", tenant_id="t1", auth_token="tok")
        assert a == b

    def test_hashable(self):
        ctx = UserContext(user_id="u1", tenant_id="t1", auth_token="tok")
        assert hash(ctx) == hash(UserContext(user_id="u1", tenant_id="t1", auth_token="tok"))

    def test_redacts_auth_token_in_repr(self):
        ctx = UserContext(user_id="u1", tenant_id="t1", auth_token="secret-token-12345")
        r = repr(ctx)
        assert "secret-token-12345" not in r
        assert "u1" in r
