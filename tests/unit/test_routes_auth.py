"""Unit tests for routes/_auth._user_ctx_from."""
from unittest.mock import MagicMock

import pytest

from src.domain.errors import AuthError
from src.domain.types import UserContext
from src.routes._auth import _user_ctx_from


def _request(headers: dict) -> object:
    """Build a minimal stand-in for fastapi.Request — only headers matter."""
    req = MagicMock()
    # FastAPI Request.headers is case-insensitive; MagicMock() with .get()
    # is good enough since we always call request.headers.get("x-...").
    req.headers = {k.lower(): v for k, v in headers.items()}
    return req


def test_extracts_email_role_and_token():
    req = _request({
        "X-User-Email": "alice@example.com",
        "X-User-Role": "manager",
        "Authorization": "Bearer hmac-token-abc",
    })
    ctx = _user_ctx_from(req)
    assert isinstance(ctx, UserContext)
    assert ctx.user_id == "alice@example.com"
    assert ctx.tenant_id == "manager"
    assert ctx.auth_token == "hmac-token-abc"


def test_role_defaults_to_default_when_absent():
    req = _request({
        "X-User-Email": "alice@example.com",
        "Authorization": "Bearer tok",
    })
    ctx = _user_ctx_from(req)
    assert ctx.tenant_id == "default"


def test_token_defaults_to_empty_when_authorization_absent():
    req = _request({"X-User-Email": "alice@example.com"})
    ctx = _user_ctx_from(req)
    assert ctx.auth_token == ""


def test_token_defaults_to_empty_when_authorization_not_bearer():
    req = _request({
        "X-User-Email": "alice@example.com",
        "Authorization": "Basic abc",
    })
    ctx = _user_ctx_from(req)
    assert ctx.auth_token == ""


def test_raises_authError_when_email_missing():
    req = _request({"X-User-Role": "manager", "Authorization": "Bearer tok"})
    with pytest.raises(AuthError, match="X-User-Email"):
        _user_ctx_from(req)


def test_email_normalised_to_lowercase_and_stripped():
    req = _request({"X-User-Email": "  Alice@Example.COM  "})
    ctx = _user_ctx_from(req)
    assert ctx.user_id == "alice@example.com"
