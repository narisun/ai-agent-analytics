"""Unit tests for analytics graph routing logic."""

from src.nodes.intent_router import route_after_intent


class TestRouteAfterIntent:
    """Test conditional edge routing based on intent classification."""

    def test_data_query_routes_to_mcp(self):
        state = {"intent": "data_query"}
        assert route_after_intent(state) == "mcp_tool_caller"

    def test_follow_up_routes_to_synthesis(self):
        state = {"intent": "follow_up"}
        assert route_after_intent(state) == "synthesis"

    def test_clarification_routes_to_error_handler(self):
        state = {"intent": "clarification"}
        assert route_after_intent(state) == "error_handler"

    def test_default_routes_to_mcp(self):
        """Unknown or missing intent defaults to data_query path."""
        state = {}
        assert route_after_intent(state) == "mcp_tool_caller"

    def test_unknown_intent_routes_to_mcp(self):
        state = {"intent": "unknown_type"}
        assert route_after_intent(state) == "error_handler"
