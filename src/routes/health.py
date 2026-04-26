"""Health endpoints — liveness + readiness."""
from __future__ import annotations

from fastapi import APIRouter, Request

health_router = APIRouter(tags=["health"])


@health_router.get("/health")
async def health() -> dict:
    """Basic liveness probe — returns 200 if the process is up."""
    return {"status": "ok"}


@health_router.get("/health/ready")
async def health_ready(request: Request) -> dict:
    """Readiness probe — checks the bridges in app.state.bridges if present.

    Compatible with both the old structure (app.state.bridges set directly
    by the legacy lifespan) and the new structure (app.state.deps populated
    by Phase 4.3's lifespan). Returns 200 with bridge status when at least
    one bridge is connected; otherwise reports 'degraded'.
    """
    bridges = getattr(request.app.state, "bridges", None)
    if bridges is None:
        # Try the new deps shape — Phase 4.3 attaches deps that may carry
        # an mcp_tools_provider rather than a raw bridges dict.
        deps = getattr(request.app.state, "deps", None)
        if deps is not None and getattr(deps, "graph", None) is not None:
            return {"status": "ready"}
        return {"status": "starting", "bridges": None}

    bridge_status = {name: bridge.is_connected for name, bridge in bridges.items()}
    all_connected = all(bridge_status.values()) if bridge_status else False
    return {
        "status": "ready" if all_connected else "degraded",
        "bridges": bridge_status,
    }
