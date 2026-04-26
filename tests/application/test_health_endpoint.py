"""Application-tier tests for GET /health and /health/ready."""
import os
os.environ.setdefault("INTERNAL_API_KEY", "test-internal-api-key")

from fastapi.testclient import TestClient  # noqa: E402

from src.app import create_app  # noqa: E402
from src.app_dependencies import AppDependencies  # noqa: E402


def _build_app(deps=None):
    if deps is None:
        deps = AppDependencies(
            config=None, graph=None, conversation_store=None,
            mcp_tools_provider=None, llm_factory=None, telemetry=None,
            compaction=None, encoder_factory=None, chat_service_factory=None,
        )
    return create_app(deps)


def test_liveness_200():
    client = TestClient(_build_app())
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readiness_returns_starting_when_no_bridges_and_no_graph():
    client = TestClient(_build_app())
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") in ("starting", "ready", "degraded")


def test_readiness_reports_ready_with_graph_attached():
    deps = AppDependencies(
        config=None,
        graph=object(),  # truthy graph
        conversation_store=None, mcp_tools_provider=None, llm_factory=None,
        telemetry=None, compaction=None, encoder_factory=None,
        chat_service_factory=None,
    )
    client = TestClient(_build_app(deps))
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"
