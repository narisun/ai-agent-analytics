"""Build a UserContext from request headers.

Header contract (matches the dashboard's existing convention; see
src/app.py:386-387 in the pre-Phase-7 inline endpoint):

    X-User-Email     required — used as UserContext.user_id
    X-User-Role      optional — used as UserContext.tenant_id; defaults to "default"
    Authorization    optional — Bearer token mapped to UserContext.auth_token

Phase 9 will replace the role-as-tenant mapping with a real tenant
claim from a verified JWT. Until then this preserves the dashboard's
existing header contract while plumbing user identity through the
new SoC/DI structure.
"""
from __future__ import annotations

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

    auth = headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        token = auth[len("bearer "):].strip()
    else:
        token = ""

    return UserContext(user_id=email, tenant_id=role, auth_token=token)
