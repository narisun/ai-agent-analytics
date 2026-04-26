"""Domain value types — frozen dataclasses and pydantic models.

UserContext replaces the module-level auth ContextVar in platform_sdk.mcp_bridge.
It flows explicitly through ChatService and into LangGraph config["configurable"].
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class UserContext:
    """Per-request user identity and auth scope.

    auth_token is never shown in repr/log output.
    """
    user_id: str
    tenant_id: str
    auth_token: str = field(repr=False)

    def __repr__(self) -> str:
        return f"UserContext(user_id={self.user_id!r}, tenant_id={self.tenant_id!r}, auth_token=<redacted>)"
