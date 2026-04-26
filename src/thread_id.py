"""Tenant-scoped LangGraph `thread_id` construction.

Derives a server-side thread key from `(user_email, session_id)` so two
users from different tenants cannot collide on the LangGraph
checkpointer even if they happen to share a session UUID. The helper is
deliberately small and pure: no I/O, no logging, no globals — every
caller and test can substitute its own `(email, session_id)` and reason
about the result locally.
"""
from typing import Final

_DEFAULT_USER: Final[str] = "anonymous"


def make_thread_id(user_email: str | None, session_id: str) -> str:
    """Return a tenant-namespaced LangGraph `thread_id`.

    The result is the literal concatenation
    ``"{normalized_email}:{session_id}"``. It is deterministic for a
    given ``(email, session_id)`` pair, and different emails always
    produce different `thread_id` values even when `session_id` is
    identical. Email is lowercased and stripped so SSO/case variations
    do not fork the same user into two threads. An empty or `None`
    email falls back to the literal namespace ``"anonymous"`` so dev
    and unauthenticated paths still get a stable, non-colliding key.

    Args:
        user_email: Authenticated user identity from the
            ``X-User-Email`` header. May be empty or ``None``.
        session_id: Client-supplied conversation thread UUID.

    Returns:
        A string suitable as the LangGraph
        ``configurable.thread_id``.

    Raises:
        ValueError: if ``session_id`` is empty.
    """
    if not session_id:
        raise ValueError("session_id is required")
    email = (user_email or "").strip().lower() or _DEFAULT_USER
    return f"{email}:{session_id}"
