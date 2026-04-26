"""Build a UserContext from request headers.

Header contract:

    X-User-Email   required — UserContext.user_id
    X-User-Role    optional — UserContext.tenant_id; defaults to "default"

UserContext.auth_token is an HMAC-signed AgentContext header value
minted server-side here. Downstream MCP servers verify the HMAC with
CONTEXT_HMAC_SECRET — the LLM cannot forge it, and the dashboard does
not need to know the secret.
"""
from __future__ import annotations

from platform_sdk import AgentContext

from ..domain.errors import AuthError
from ..domain.types import UserContext


def _user_ctx_from(request) -> UserContext:
    """Extract a UserContext from a FastAPI Request.

    Raises AuthError when X-User-Email is missing. Email is lowercased
    and stripped so SSO/case variants do not fork the same user across
    tenant-scoped stores.
    """
    headers = request.headers
    email_raw = headers.get("x-user-email") or ""
    email = email_raw.strip().lower()
    if not email:
        raise AuthError("Missing X-User-Email header")

    role = headers.get("x-user-role") or "default"

    per_request_ctx = AgentContext(
        rm_id=email,
        rm_name=email.split("@")[0] if "@" in email else email,
        role="manager",
        team_id="analytics",
        assigned_account_ids=(),
        compliance_clearance=("standard", "aml_view"),
        user_email=email,
        user_role=role,
    )
    token = per_request_ctx.to_header_value()

    return UserContext(user_id=email, tenant_id=role, auth_token=token)
