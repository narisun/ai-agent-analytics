"""Unit tests for the tenant-scoped LangGraph thread_id helper."""
import pytest

from src.thread_id import make_thread_id


class TestMakeThreadId:
    """Contract for `make_thread_id(user_email, session_id) -> str`."""

    def test_concatenates_email_and_session_with_colon(self):
        assert make_thread_id("alice@example.com", "abc-123") == "alice@example.com:abc-123"

    def test_normalizes_email_to_lowercase(self):
        # SSO providers and email clients vary on case; the namespace must
        # be insensitive to them so two requests for the same person never
        # accidentally fork into two threads.
        assert make_thread_id("Alice@Example.COM", "abc-123") == "alice@example.com:abc-123"

    def test_strips_surrounding_whitespace_in_email(self):
        assert make_thread_id("  alice@example.com  ", "abc-123") == "alice@example.com:abc-123"

    def test_falls_back_to_anonymous_namespace_when_email_empty(self):
        # Empty / None X-User-Email defaults to a stable "anonymous" namespace
        # so dev and unauthenticated paths get a consistent thread key,
        # never a thread shared across users.
        assert make_thread_id("", "abc-123") == "anonymous:abc-123"
        assert make_thread_id(None, "abc-123") == "anonymous:abc-123"

    def test_different_emails_produce_different_thread_ids(self):
        same_session = "abc-123"
        alice = make_thread_id("alice@example.com", same_session)
        bob = make_thread_id("bob@example.com", same_session)
        assert alice != bob, (
            "Two users sharing a session_id MUST get distinct thread_ids — "
            "this is the core tenant-isolation guarantee."
        )

    def test_raises_on_empty_session_id(self):
        with pytest.raises(ValueError, match="session_id is required"):
            make_thread_id("alice@example.com", "")
