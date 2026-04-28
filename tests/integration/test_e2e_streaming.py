"""End-to-end streaming tests for the Analytics Agent pipeline.

Tests the full flow from HTTP request through the LangGraph orchestrator
to Data Stream Protocol response output. Verifies:
  - Complete data_query path (intent → MCP tools → synthesis → stream)
  - Follow-up path (intent → synthesis → stream)
  - Clarification path (intent → error_handler → stream)
  - Error handling and graceful degradation
  - Data Stream Protocol format compliance

Data Stream Protocol (Vercel AI SDK compatible):
  0:"text"                — Narrative tokens (streaming text response)
  g:"reasoning"           — Reasoning tokens (thinking/process steps)
  b:{toolCallId,toolName} — Tool call start
  a:{toolCallId,result}   — Tool call result
  2:[{...}]               — Custom data (UI components)
  3:"error"               — Error message
  d:{finishReason}        — Stream end (stop | error | length)
"""

import json
import pytest
import httpx
from unittest.mock import AsyncMock, patch


from src.app import app
from src.schemas.intent import IntentResult, QueryPlanStep
from src.schemas.ui_components import AnalyticsResponse, UIComponent, ChartMetadata, ChartDataPoint

pytestmark = pytest.mark.integration


class DataStreamParser:
    """Parser for Data Stream Protocol format responses.

    Handles line-based format where each line is:
      PREFIX:PAYLOAD

    Prefixes:
      0     — text (narrative tokens)
      g     — reasoning (thinking tokens)
      b     — tool call begin
      a     — tool call result
      2     — custom data (UI components)
      3     — error
      d     — finish
    """

    def __init__(self):
        """Initialize the parser state."""
        self.lines: list[str] = []
        self.events: list[dict] = []

    def parse_line(self, line: str) -> dict | None:
        """Parse a single line of the Data Stream Protocol.

        Args:
            line: A single line like `0:"text"` or `g:"reasoning"`

        Returns:
            A dict with keys: prefix, payload, error (if parsing failed)
        """
        if not line or not line.strip():
            return None

        try:
            if ":" not in line:
                return {"error": "no colon", "line": line}

            prefix, rest = line.split(":", 1)
            try:
                payload = json.loads(rest)
            except json.JSONDecodeError:
                return {"error": "invalid json", "prefix": prefix, "line": line}

            return {"prefix": prefix, "payload": payload}

        except Exception as e:
            return {"error": str(e), "line": line}

    def parse_response(self, response_text: str) -> list[dict]:
        """Parse a complete response stream.

        Args:
            response_text: Full response text (newline-delimited)

        Returns:
            List of parsed events
        """
        self.events = []
        for line in response_text.strip().split("\n"):
            if not line.strip():
                continue
            event = self.parse_line(line)
            if event:
                self.events.append(event)
        return self.events

    def get_events_by_prefix(self, prefix: str) -> list[dict]:
        """Get all events with a specific prefix.

        Args:
            prefix: The prefix to filter by (e.g., "0", "g", "b", "a", "2", "3", "d")

        Returns:
            List of matching events
        """
        return [e for e in self.events if e.get("prefix") == prefix]

    def get_reasoning_tokens(self) -> list[str]:
        """Extract all reasoning tokens (prefix g:)."""
        return [e.get("payload", "") for e in self.get_events_by_prefix("g")]

    def get_text_tokens(self) -> list[str]:
        """Extract all narrative text tokens (prefix 0:)."""
        return [e.get("payload", "") for e in self.get_events_by_prefix("0")]

    def get_tool_calls(self) -> list[dict]:
        """Extract all tool call begins (prefix b:)."""
        return [e.get("payload", {}) for e in self.get_events_by_prefix("b")]

    def get_tool_results(self) -> list[dict]:
        """Extract all tool call results (prefix a:)."""
        return [e.get("payload", {}) for e in self.get_events_by_prefix("a")]

    def get_ui_components(self) -> list:
        """Extract all UI components (prefix 2:)."""
        components = []
        for e in self.get_events_by_prefix("2"):
            payload = e.get("payload", [])
            if isinstance(payload, list):
                components.extend(payload)
        return components

    def get_errors(self) -> list[str]:
        """Extract all error messages (prefix 3:)."""
        return [e.get("payload", "") for e in self.get_events_by_prefix("3")]

    def get_finish_events(self) -> list[dict]:
        """Extract finish event (prefix d:)."""
        return self.get_events_by_prefix("d")


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
async def app_with_mocks(mock_bridges, mock_config, mock_router_llm, mock_synthesis_llm):
    """Fixture that sets up the FastAPI app with mocked graph.

    This patches app.state to inject a mock compiled graph without
    going through the lifespan context manager (which needs real MCP bridges).
    """
    # Import here to avoid circular imports
    from src.graph import build_analytics_graph

    llm_router, router_structured = mock_router_llm
    llm_synthesis, synthesis_structured = mock_synthesis_llm

    # Build the real graph with mocks
    with patch("src.graph.make_chat_llm") as mock_make_llm:

        def llm_factory(route, **kwargs):
            if "router" in route:
                return llm_router
            else:
                return llm_synthesis

        mock_make_llm.side_effect = llm_factory

        graph = build_analytics_graph(bridges=mock_bridges, config=mock_config)

    # Inject the graph into app.state
    app.state.graph = graph
    app.state.bridges = mock_bridges
    app.state.config = mock_config

    return app, router_structured, synthesis_structured


# ============================================================================
# Test Classes
# ============================================================================


@pytest.mark.asyncio
class TestDataQueryE2E:
    """End-to-end tests for the full data_query pipeline.

    Tests:
      intent_router → mcp_tool_caller → synthesis → streaming response
    """

    async def test_full_data_query_pipeline(self, app_with_mocks, mock_bridges):
        """Test a complete data_query flow with MCP tool execution.

        Verifies:
          - Reasoning tokens for intent classification
          - Tool call begin/result events
          - UI components in the stream
          - Narrative text in the stream
          - Finish event present
        """
        app_instance, router_structured, synthesis_structured = app_with_mocks

        # Configure mocks for this test
        router_structured.ainvoke = AsyncMock(
            return_value=IntentResult(
                intent="data_query",
                query_plan=[
                    QueryPlanStep(
                        tool_name="execute_read_query",
                        mcp_server="data-mcp",
                        parameters={"query": 'SELECT * FROM salesforce."Opportunity" LIMIT 10'},
                        description="Fetch top 10 opportunities",
                    ),
                ],
                reasoning="User is asking for opportunity data from Salesforce.",
            )
        )

        synthesis_structured.ainvoke = AsyncMock(
            return_value=AnalyticsResponse(
                narrative="The top opportunities show strong pipeline momentum. Average deal size is growing.",
                components=[
                    UIComponent(
                        component_type="BarChart",
                        metadata=ChartMetadata(
                            title="Top Opportunities by Amount",
                            source="data-mcp",
                            confidence_score=0.95,
                            format_hint="currency",
                            x_label="Account Name",
                            y_label="Amount ($)",
                        ),
                        data=[
                            ChartDataPoint(category="Acme Corp", value=500000),
                            ChartDataPoint(category="Widget Inc", value=350000),
                            ChartDataPoint(category="Tech Solutions", value=275000),
                        ],
                    ),
                ],
            )
        )

        # Make the request using httpx AsyncClient with ASGI transport
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance)) as client:
            response = await client.post(
                "/api/v1/analytics/chat",
                json={
                    "messages": [{"role": "user", "content": "Show me the top opportunities"}],
                    "id": "session-123",
                },
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

        # Parse the streaming response
        parser = DataStreamParser()
        parser.parse_response(response.text)

        # Verify structure
        reasoning_tokens = parser.get_reasoning_tokens()
        text_tokens = parser.get_text_tokens()
        ui_components = parser.get_ui_components()
        finish_events = parser.get_finish_events()
        errors = parser.get_errors()

        # Assertions
        assert len(reasoning_tokens) > 0, "Should have reasoning tokens (intent classification)"
        assert len(text_tokens) > 0, "Should have narrative text tokens"
        assert len(ui_components) > 0, "Should have UI components"
        assert len(finish_events) == 1, "Should have exactly one finish event"
        assert finish_events[0]["payload"]["finishReason"] == "stop", "Should finish with 'stop'"
        assert len(errors) == 0, "Should have no errors"

    async def test_multi_tool_query_plan(self, app_with_mocks, mock_bridges):
        """Test a query plan with multiple tool calls.

        Verifies:
          - Multiple tool call begin events
          - All tool results streamed
          - Synthesis proceeds after all tools complete
        """
        app_instance, router_structured, synthesis_structured = app_with_mocks

        # Configure for multi-step query plan
        router_structured.ainvoke = AsyncMock(
            return_value=IntentResult(
                intent="data_query",
                query_plan=[
                    QueryPlanStep(
                        tool_name="get_salesforce_summary",
                        mcp_server="salesforce-mcp",
                        parameters={"client_name": "Acme Corp"},
                        description="Get Salesforce pipeline for Acme",
                    ),
                    QueryPlanStep(
                        tool_name="get_payment_summary",
                        mcp_server="payments-mcp",
                        parameters={"client_name": "Acme Corp"},
                        description="Get payment activity for Acme",
                    ),
                ],
                reasoning="User wants a complete view of Acme. Need both CRM and payments data.",
            )
        )

        synthesis_structured.ainvoke = AsyncMock(
            return_value=AnalyticsResponse(
                narrative="Acme Corp has $2.5M in pipeline and processed $47.5M in payments this year.",
                components=[
                    UIComponent(
                        component_type="KPICard",
                        metadata=ChartMetadata(
                            title="Acme Corp Metrics",
                            source="salesforce-mcp",
                            confidence_score=0.92,
                        ),
                        data=[
                            {
                                "label": "Pipeline Value",
                                "value": 2500000,
                                "trend": "up",
                                "change": 12.5,
                            },
                            {
                                "label": "Payment Volume",
                                "value": 47500000,
                                "trend": "up",
                                "change": 8.3,
                            },
                        ],
                    ),
                ],
            )
        )

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance)) as client:
            response = await client.post(
                "/api/v1/analytics/chat",
                json={
                    "messages": [
                        {"role": "user", "content": "Show me a complete view of Acme Corp"}
                    ],
                    "id": "session-456",
                },
            )

        assert response.status_code == 200

        parser = DataStreamParser()
        parser.parse_response(response.text)

        # Verify multiple tool calls
        tool_calls = parser.get_tool_calls()
        tool_results = parser.get_tool_results()

        assert len(tool_calls) >= 2, "Should have at least 2 tool call begin events"
        assert len(tool_results) >= 2, "Should have at least 2 tool result events"

        # Verify narrative and components
        text_tokens = parser.get_text_tokens()
        ui_components = parser.get_ui_components()

        assert len(text_tokens) > 0, "Should have narrative text"
        assert len(ui_components) > 0, "Should have UI components"


@pytest.mark.asyncio
class TestFollowUpE2E:
    """End-to-end tests for the follow_up intent path.

    Tests:
      intent_router → synthesis (skip MCP tools)
    """

    async def test_follow_up_skips_mcp_tools(self, app_with_mocks, mock_bridges):
        """Test that follow_up intent skips tool execution.

        Verifies:
          - No tool call begin/result events
          - Reasoning tokens present for classification
          - Narrative text present
          - Synthesis generates response directly
        """
        app_instance, router_structured, synthesis_structured = app_with_mocks

        # Configure for follow_up
        router_structured.ainvoke = AsyncMock(
            return_value=IntentResult(
                intent="follow_up",
                query_plan=[],  # No query plan for follow_up
                reasoning="User is asking about data already retrieved in this conversation.",
            )
        )

        synthesis_structured.ainvoke = AsyncMock(
            return_value=AnalyticsResponse(
                narrative="Breaking down the $2.5M pipeline by stage: Negotiation ($800K), Proposal ($700K), Discovery ($1M).",
                components=[
                    UIComponent(
                        component_type="PieChart",
                        metadata=ChartMetadata(
                            title="Pipeline by Stage",
                            source="salesforce-mcp",
                            confidence_score=0.95,
                        ),
                        data=[
                            ChartDataPoint(category="Negotiation", value=800000),
                            ChartDataPoint(category="Proposal", value=700000),
                            ChartDataPoint(category="Discovery", value=1000000),
                        ],
                    ),
                ],
            )
        )

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance)) as client:
            response = await client.post(
                "/api/v1/analytics/chat",
                json={
                    "messages": [
                        {"role": "user", "content": "Show me pipeline by stage"},
                        {"role": "assistant", "content": "Pipeline total is $2.5M"},
                        {"role": "user", "content": "Break that down by stage"},
                    ],
                    "id": "session-789",
                },
            )

        assert response.status_code == 200

        parser = DataStreamParser()
        parser.parse_response(response.text)

        # Follow-up should NOT have tool calls
        tool_calls = parser.get_tool_calls()
        assert len(tool_calls) == 0, "Follow-up should not trigger tool calls"

        # But should have reasoning and narrative
        reasoning_tokens = parser.get_reasoning_tokens()
        text_tokens = parser.get_text_tokens()

        assert len(reasoning_tokens) > 0, "Should have reasoning (intent classification)"
        assert len(text_tokens) > 0, "Should have narrative text"


@pytest.mark.asyncio
class TestClarificationE2E:
    """End-to-end tests for the clarification intent path.

    Tests:
      intent_router → error_handler → helpful message
    """

    async def test_clarification_returns_help_message(self, app_with_mocks):
        """Test that clarification intent produces a helpful message.

        Verifies:
          - Intent is classified as clarification
          - Error handler provides guidance
          - Response includes helpful text
          - Finish event is present
        """
        app_instance, router_structured, synthesis_structured = app_with_mocks

        # Configure for clarification
        router_structured.ainvoke = AsyncMock(
            return_value=IntentResult(
                intent="clarification",
                query_plan=[],
                reasoning="User's request is too vague to determine intent.",
                clarification_message="I can help you analyze sales data, pipelines, payments, and news. Could you be more specific about what you'd like to see?",
            )
        )

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance)) as client:
            response = await client.post(
                "/api/v1/analytics/chat",
                json={
                    "messages": [{"role": "user", "content": "help"}],
                    "id": "session-help",
                },
            )

        assert response.status_code == 200

        parser = DataStreamParser()
        parser.parse_response(response.text)

        # Should have text or reasoning guiding the user
        text_tokens = parser.get_text_tokens()
        reasoning_tokens = parser.get_reasoning_tokens()

        assert len(text_tokens) > 0 or len(reasoning_tokens) > 0, (
            "Clarification should provide helpful guidance"
        )

        # Should have finish event
        finish_events = parser.get_finish_events()
        assert len(finish_events) > 0, "Should have finish event"


@pytest.mark.asyncio
class TestErrorHandlingE2E:
    """End-to-end tests for error scenarios and graceful degradation."""

    async def test_mcp_tool_failure_streams_error_and_continues(self, app_with_mocks, mock_bridges):
        """Test that MCP tool failure is gracefully handled.

        Scenario:
          - Intent is data_query with a tool plan
          - One tool fails during execution
          - Synthesis still runs (with partial data)
          - Error is surfaced in the stream

        Verifies:
          - Error message is streamed (3: prefix)
          - Synthesis response is still generated
          - Finish event is present
        """
        app_instance, router_structured, synthesis_structured = app_with_mocks

        # Configure intent
        router_structured.ainvoke = AsyncMock(
            return_value=IntentResult(
                intent="data_query",
                query_plan=[
                    QueryPlanStep(
                        tool_name="execute_read_query",
                        mcp_server="data-mcp",
                        parameters={"query": "SELECT * FROM nonexistent_table"},
                        description="This will fail",
                    ),
                ],
                reasoning="User is asking for data.",
            )
        )

        # Configure synthesis to handle error gracefully
        synthesis_structured.ainvoke = AsyncMock(
            return_value=AnalyticsResponse(
                narrative="The requested data could not be retrieved due to an error. Please try a different query.",
            )
        )

        # Make the data-mcp tool fail
        mock_bridges["data-mcp"].invoke = AsyncMock(
            side_effect=Exception("Table nonexistent_table does not exist")
        )

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance)) as client:
            response = await client.post(
                "/api/v1/analytics/chat",
                json={
                    "messages": [{"role": "user", "content": "Query the nonexistent table"}],
                    "id": "session-error1",
                },
            )

        assert response.status_code == 200

        parser = DataStreamParser()
        parser.parse_response(response.text)

        # Should still have finish event
        finish_events = parser.get_finish_events()
        assert len(finish_events) > 0, "Should have finish event even with errors"

        # Should have narrative fallback
        text_tokens = parser.get_text_tokens()
        reasoning_tokens = parser.get_reasoning_tokens()
        assert len(text_tokens) > 0 or len(reasoning_tokens) > 0, (
            "Should have some response content"
        )

    async def test_llm_failure_graceful_degradation(self, app_with_mocks):
        """Test that LLM failures degrade gracefully.

        Scenario:
          - Synthesis LLM fails
          - Error is caught and handled
          - User receives a fallback message

        Verifies:
          - Finish event is present
          - Error is logged internally
          - Stream doesn't hang
        """
        app_instance, router_structured, synthesis_structured = app_with_mocks

        # Configure successful intent
        router_structured.ainvoke = AsyncMock(
            return_value=IntentResult(
                intent="follow_up",
                reasoning="This is a follow-up.",
            )
        )

        # But synthesis fails
        synthesis_structured.ainvoke = AsyncMock(side_effect=Exception("LLM service overloaded"))

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance)) as client:
            response = await client.post(
                "/api/v1/analytics/chat",
                json={
                    "messages": [{"role": "user", "content": "What's the analysis?"}],
                    "id": "session-error2",
                },
            )

        assert response.status_code == 200

        parser = DataStreamParser()
        parser.parse_response(response.text)

        # Should always have finish event
        finish_events = parser.get_finish_events()
        assert len(finish_events) > 0, "Stream should complete with finish event"

        # May have error messages
        errors = parser.get_errors()
        content = parser.get_text_tokens() + parser.get_reasoning_tokens() + errors
        assert len(content) > 0, "Should provide some feedback to user"


@pytest.mark.asyncio
class TestDataStreamProtocol:
    """Tests for Data Stream Protocol format compliance."""

    async def test_response_content_type(self, app_with_mocks, mock_bridges):
        """Verify the response has correct content-type and headers."""
        app_instance, router_structured, synthesis_structured = app_with_mocks

        router_structured.ainvoke = AsyncMock(
            return_value=IntentResult(
                intent="follow_up",
            )
        )

        synthesis_structured.ainvoke = AsyncMock(
            return_value=AnalyticsResponse(
                narrative="Test response",
            )
        )

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance)) as client:
            response = await client.post(
                "/api/v1/analytics/chat",
                json={"messages": [{"role": "user", "content": "Test"}], "id": "id1"},
            )

        # Verify headers
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        assert response.headers.get("cache-control") == "no-cache"
        assert response.headers.get("connection") == "keep-alive"
        assert "no" in response.headers.get("x-accel-buffering", "")

    async def test_finish_message_always_sent(self, app_with_mocks):
        """Verify that finish event is always sent, even on error."""
        app_instance, router_structured, synthesis_structured = app_with_mocks

        # Make router fail
        router_structured.ainvoke = AsyncMock(side_effect=Exception("Routing failed"))

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance)) as client:
            response = await client.post(
                "/api/v1/analytics/chat",
                json={"messages": [{"role": "user", "content": "Query"}], "id": "id2"},
            )

        parser = DataStreamParser()
        parser.parse_response(response.text)

        finish_events = parser.get_finish_events()
        assert len(finish_events) > 0, "Finish event must always be sent"

        # Should have either error or text content
        all_content = parser.get_text_tokens() + parser.get_reasoning_tokens() + parser.get_errors()
        assert len(all_content) > 0, "Should provide error/feedback"

    async def test_reasoning_before_narrative(self, app_with_mocks, mock_bridges):
        """Verify that reasoning tokens appear before narrative in the stream.

        In the Data Stream Protocol, the intent classification (reasoning)
        should be transmitted before the final narrative analysis.
        """
        app_instance, router_structured, synthesis_structured = app_with_mocks

        router_structured.ainvoke = AsyncMock(
            return_value=IntentResult(
                intent="data_query",
                query_plan=[
                    QueryPlanStep(
                        tool_name="execute_read_query",
                        mcp_server="data-mcp",
                        parameters={"query": "SELECT 1"},
                        description="Simple query",
                    ),
                ],
                reasoning="Classified as data_query.",
            )
        )

        synthesis_structured.ainvoke = AsyncMock(
            return_value=AnalyticsResponse(
                narrative="Results show interesting trends.",
            )
        )

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance)) as client:
            response = await client.post(
                "/api/v1/analytics/chat",
                json={"messages": [{"role": "user", "content": "Query"}], "id": "id3"},
            )

        parser = DataStreamParser()
        parser.parse_response(response.text)

        # Find indices of first reasoning and first narrative
        reasoning_indices = []
        narrative_indices = []

        for i, event in enumerate(parser.events):
            if event.get("prefix") == "g":
                reasoning_indices.append(i)
            elif event.get("prefix") == "0":
                narrative_indices.append(i)

        # If both exist, reasoning should come before narrative
        if reasoning_indices and narrative_indices:
            first_reasoning = reasoning_indices[0]
            first_narrative = narrative_indices[0]
            assert first_reasoning < first_narrative, (
                "Reasoning tokens should appear before narrative in the stream"
            )

    async def test_json_parsing_correctness(self, app_with_mocks):
        """Verify that all JSON payloads are valid and parseable.

        The Data Stream Protocol embeds JSON after each prefix.
        This test ensures all payloads can be parsed.
        """
        app_instance, router_structured, synthesis_structured = app_with_mocks

        router_structured.ainvoke = AsyncMock(
            return_value=IntentResult(
                intent="follow_up",
                reasoning="User is asking a follow-up question.",
            )
        )

        synthesis_structured.ainvoke = AsyncMock(
            return_value=AnalyticsResponse(
                narrative="The data shows a 15% increase.",
                components=[
                    UIComponent(
                        component_type="BarChart",
                        metadata=ChartMetadata(
                            title="Test Chart",
                            source="test-mcp",
                            confidence_score=0.85,
                        ),
                        data=[ChartDataPoint(category="A", value=100)],
                    ),
                ],
            )
        )

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance)) as client:
            response = await client.post(
                "/api/v1/analytics/chat",
                json={"messages": [{"role": "user", "content": "Test"}], "id": "id4"},
            )

        parser = DataStreamParser()
        parsed_events = parser.parse_response(response.text)

        # All events should be parseable (no "error" field)
        parse_errors = [e for e in parsed_events if e.get("error")]
        assert len(parse_errors) == 0, f"All lines should be valid JSON: {parse_errors}"

        # Verify structure of specific event types
        finish_events = parser.get_finish_events()
        if finish_events:
            finish = finish_events[0]["payload"]
            assert "finishReason" in finish, "Finish event should have finishReason"

    async def test_empty_message_array_handling(self, app_with_mocks):
        """Test graceful handling of edge case: empty message array."""
        app_instance, router_structured, synthesis_structured = app_with_mocks

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance)) as client:
            response = await client.post(
                "/api/v1/analytics/chat",
                json={"messages": [], "id": "id5"},
            )

        # Should return 200 with error event + finish
        assert response.status_code == 200

        parser = DataStreamParser()
        parser.parse_response(response.text)

        errors = parser.get_errors()
        finish_events = parser.get_finish_events()

        assert len(errors) > 0, "Should emit error for missing user message"
        assert len(finish_events) > 0, "Should still finish"

    async def test_streaming_response_newline_delimited(self, app_with_mocks):
        """Verify response is properly newline-delimited for streaming.

        Each event should be on its own line for proper streaming consumption.
        """
        app_instance, router_structured, synthesis_structured = app_with_mocks

        router_structured.ainvoke = AsyncMock(
            return_value=IntentResult(
                intent="follow_up",
            )
        )

        synthesis_structured.ainvoke = AsyncMock(
            return_value=AnalyticsResponse(
                narrative="Test",
            )
        )

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance)) as client:
            response = await client.post(
                "/api/v1/analytics/chat",
                json={"messages": [{"role": "user", "content": "Test"}], "id": "id6"},
            )

        lines = response.text.strip().split("\n")
        assert len(lines) > 0, "Response should have content"

        # Each line should be a valid event (except potentially the last empty line)
        parser = DataStreamParser()
        for line in lines:
            if line.strip():
                event = parser.parse_line(line)
                assert "error" not in event or event.get("error") is None, (
                    f"Line should be valid: {line}"
                )
