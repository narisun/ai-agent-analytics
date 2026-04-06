"""Per-session rate limiting middleware using sliding window algorithm.

Tracks requests per session_id with configurable limit (default: 20 req/min).
Uses in-memory sliding window with periodic cleanup of stale sessions.
Returns HTTP 429 with Retry-After header when limit exceeded.
"""
import os
import time
from collections import defaultdict
from typing import Optional

from fastapi import Depends, HTTPException, Request
from platform_sdk import get_logger

log = get_logger(__name__)


# Configuration from environment
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "20"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
CLEANUP_INTERVAL_SECONDS = 300  # Clean up stale sessions every 5 minutes


class RateLimiter:
    """In-memory sliding window rate limiter.

    Tracks request timestamps per session_id. Cleans up stale sessions
    periodically to prevent memory growth.
    """

    def __init__(self, rpm_limit: int = RATE_LIMIT_RPM, window_seconds: int = RATE_LIMIT_WINDOW_SECONDS):
        """Initialize rate limiter.

        Args:
            rpm_limit: Requests per minute (default: 20)
            window_seconds: Sliding window duration in seconds (default: 60)
        """
        self.rpm_limit = rpm_limit
        self.window_seconds = window_seconds
        self.sessions: dict[str, list[float]] = defaultdict(list)
        self.last_cleanup = time.time()

    def _cleanup_stale_sessions(self) -> None:
        """Remove sessions with no recent activity.

        A session is considered stale if all timestamps are outside
        the current window. This prevents unbounded memory growth.
        """
        now = time.time()
        if now - self.last_cleanup < CLEANUP_INTERVAL_SECONDS:
            return

        cutoff_time = now - self.window_seconds
        stale_sessions = [
            session_id
            for session_id, timestamps in self.sessions.items()
            if not any(ts > cutoff_time for ts in timestamps)
        ]

        for session_id in stale_sessions:
            del self.sessions[session_id]
            log.debug("rate_limiter_cleanup", session_id=session_id)

        self.last_cleanup = now

    def is_allowed(self, session_id: str) -> tuple[bool, Optional[int]]:
        """Check if request is allowed under rate limit.

        Args:
            session_id: Unique session identifier

        Returns:
            Tuple of (allowed: bool, retry_after_seconds: Optional[int])
            If allowed=False, retry_after_seconds is the minimum wait time.
        """
        self._cleanup_stale_sessions()

        now = time.time()
        cutoff_time = now - self.window_seconds

        # Keep only recent timestamps within the sliding window
        self.sessions[session_id] = [ts for ts in self.sessions[session_id] if ts > cutoff_time]

        # Check if we're at or over the limit
        request_count = len(self.sessions[session_id])
        if request_count >= self.rpm_limit:
            # Calculate how long to wait until the oldest request falls out of window
            oldest_timestamp = self.sessions[session_id][0]
            retry_after = int(self.window_seconds - (now - oldest_timestamp)) + 1
            log.warning(
                "rate_limit_exceeded",
                session_id=session_id,
                count=request_count,
                limit=self.rpm_limit,
            )
            return False, retry_after

        # Record this request and allow it
        self.sessions[session_id].append(now)
        log.debug(
            "rate_limit_allowed",
            session_id=session_id,
            count=request_count + 1,
            limit=self.rpm_limit,
        )
        return True, None


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def make_rate_limiter(
    rpm_limit: int = RATE_LIMIT_RPM, window_seconds: int = RATE_LIMIT_WINDOW_SECONDS
):
    """Factory function to create a rate limiting dependency.

    Args:
        rpm_limit: Requests per minute (default: from RATE_LIMIT_RPM env var)
        window_seconds: Sliding window duration (default: from RATE_LIMIT_WINDOW_SECONDS env var)

    Returns:
        A FastAPI dependency function that checks rate limits
    """

    def rate_limit_dependency(request: Request) -> None:
        """FastAPI dependency that enforces per-session rate limits.

        Extracts session_id from request body (for POST with ChatRequest)
        or from query parameters. Returns 429 if rate limit exceeded.

        Args:
            request: FastAPI Request object

        Raises:
            HTTPException: 429 if rate limit exceeded
        """
        global _rate_limiter

        if _rate_limiter is None:
            _rate_limiter = RateLimiter(rpm_limit=rpm_limit, window_seconds=window_seconds)

        # Extract session_id from request
        session_id = _extract_session_id(request)

        if not session_id:
            log.warning("rate_limiter_no_session_id")
            raise HTTPException(status_code=400, detail="Missing session_id in request")

        allowed, retry_after = _rate_limiter.is_allowed(session_id)

        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {rpm_limit} requests per {window_seconds} seconds",
                headers={"Retry-After": str(retry_after)},
            )

    return rate_limit_dependency


async def _extract_session_id(request: Request) -> Optional[str]:
    """Extract session_id from request body or query parameters.

    Args:
        request: FastAPI Request object

    Returns:
        session_id string or None if not found
    """
    try:
        # Try to extract from query parameters first
        session_id = request.query_params.get("session_id")
        if session_id:
            return session_id

        # Try to extract from POST body (ChatRequest or VercelChatRequest)
        if request.method == "POST":
            # Need to read and re-stream the body
            body = await request.body()
            if body:
                import json

                try:
                    data = json.loads(body)
                    # ChatRequest uses session_id directly
                    if "session_id" in data:
                        return data["session_id"]
                    # VercelChatRequest uses 'id' field for session tracking
                    if "id" in data:
                        return data["id"]
                except json.JSONDecodeError:
                    pass

    except Exception as e:
        log.warning("rate_limiter_extract_session_id_error", error=str(e))

    return None
