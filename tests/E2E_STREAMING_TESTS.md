# End-to-End Streaming Tests for Analytics Agent

## Overview

This document describes the comprehensive end-to-end streaming test suite for the Analytics Agent pipeline. The tests verify the full flow from HTTP request through the LangGraph orchestrator to Data Stream Protocol response output.

## Files

- **`conftest.py`** — Shared pytest fixtures for all tests
- **`test_e2e_streaming.py`** — Full end-to-end streaming test suite

## Quick Start

### Running All Tests

```bash
pytest tests/test_e2e_streaming.py -v
```

### Running Specific Test Class

```bash
pytest tests/test_e2e_streaming.py::TestDataQueryE2E -v
pytest tests/test_e2e_streaming.py::TestDataStreamProtocol -v
```

### Running Single Test

```bash
pytest tests/test_e2e_streaming.py::TestDataQueryE2E::test_full_data_query_pipeline -v
```

## Fixtures (conftest.py)

### `mock_bridges`
A dictionary of mock MCP bridges with realistic tool schemas:
- **data-mcp**: Database query execution tool
- **salesforce-mcp**: Salesforce CRM data tools
- **payments-mcp**: Payment and transaction tools
- **news-search-mcp**: Company news search tool

Each bridge:
- Has `is_connected = True`
- Implements `get_langchain_tools()` returning realistic tool schemas
- Implements `invoke()` for tool execution with mock responses

### `mock_config`
An AgentConfig-like mock with:
- `router_model_route = "gpt-4o-mini"` (fast intent classification)
- `synthesis_model_route = "gpt-4o"` (complex analysis)
- `mcp_startup_timeout = 30`
- Additional routing and timeout parameters

### `mock_router_llm` / `mock_router_llm_factory`
Mock LLMs configured to return structured `IntentResult` objects:
- Intent classification: "data_query" | "follow_up" | "clarification"
- Query plan generation for data retrieval
- Reasoning chain-of-thought

### `mock_synthesis_llm` / `mock_synthesis_llm_factory`
Mock LLMs configured to return structured `AnalyticsResponse` objects:
- Narrative summaries (2-3 sentence analysis)
- UI components (charts, KPIs, tables)

## Test Classes and Methods

### TestDataQueryE2E (2 tests)
Tests the full data_query pipeline: `intent_router → mcp_tool_caller → synthesis → stream`

#### `test_full_data_query_pipeline`
- Verifies complete data retrieval and analysis flow
- Confirms reasoning tokens for intent classification
- Confirms tool call begin/result events
- Confirms UI components streamed
- Confirms narrative text present
- Confirms finish event sent

#### `test_multi_tool_query_plan`
- Tests query plan with multiple tool calls (e.g., Salesforce + Payments)
- Verifies all tool calls are streamed
- Confirms synthesis proceeds with all data

### TestFollowUpE2E (1 test)
Tests the follow_up intent path: `intent_router → synthesis` (skip MCP tools)

#### `test_follow_up_skips_mcp_tools`
- Verifies no tool call events are emitted
- Confirms reasoning tokens for classification
- Confirms narrative generated directly from cached data

### TestClarificationE2E (1 test)
Tests the clarification intent path: `intent_router → error_handler → helpful message`

#### `test_clarification_returns_help_message`
- Verifies ambiguous requests get helpful guidance
- Confirms stream completes successfully

### TestErrorHandlingE2E (2 tests)
Tests error scenarios and graceful degradation

#### `test_mcp_tool_failure_streams_error_and_continues`
- One MCP tool fails during execution
- Synthesis still runs with partial data
- Error is surfaced in stream
- Confirms graceful degradation

#### `test_llm_failure_graceful_degradation`
- LLM service failure (e.g., overloaded)
- Error is caught and handled
- Stream doesn't hang

### TestDataStreamProtocol (6 tests)
Tests Data Stream Protocol format compliance

#### `test_response_content_type`
- Verifies `Content-Type: text/plain; charset=utf-8`
- Confirms cache control and connection headers
- Confirms Vercel AI SDK headers

#### `test_finish_message_always_sent`
- Verifies finish event (d:) is always sent
- Even on errors, stream properly terminates

#### `test_reasoning_before_narrative`
- Reasoning tokens (g:) appear before narrative (0:)
- Follows protocol ordering expectations

#### `test_json_parsing_correctness`
- All JSON payloads are valid and parseable
- Verifies finish event structure

#### `test_empty_message_array_handling`
- Graceful handling of edge case (no user message)
- Error event emitted with finish event

#### `test_streaming_response_newline_delimited`
- Each event is properly newline-delimited
- Confirms proper format for streaming consumption

## Data Stream Protocol Format

The response uses the Vercel AI SDK Data Stream Protocol with line-based events:

```
PREFIX:JSON_PAYLOAD
```

### Prefixes

| Prefix | Type | Example | Use Case |
|--------|------|---------|----------|
| `0` | Text | `0:"Revenue grew 15%"` | Narrative tokens (final response) |
| `g` | Reasoning | `g:"Classifying intent..."` | Thinking/process tokens |
| `b` | Tool begin | `b:{"toolCallId":"tc_xxx","toolName":"get_data"}` | Tool call start |
| `a` | Tool result | `a:{"toolCallId":"tc_xxx","result":"..."}` | Tool result data |
| `2` | Data | `2:[{component}]` | UI components (charts, KPIs, tables) |
| `3` | Error | `3:"An error occurred"` | Error messages |
| `d` | Finish | `d:{"finishReason":"stop","usage":{...}}` | Stream complete |

### Response Structure

Typical response order:
1. Intent classification (g: reasoning tokens)
2. Tool calls (b: begin, a: result pairs)
3. UI components (2: data arrays)
4. Narrative synthesis (0: text tokens)
5. Finish (d: completion marker)

## DataStreamParser Utility

The test suite includes a `DataStreamParser` class for parsing responses:

```python
parser = DataStreamParser()
parser.parse_response(response_text)

# Extract specific event types
reasoning = parser.get_reasoning_tokens()
narrative = parser.get_text_tokens()
components = parser.get_ui_components()
finish = parser.get_finish_events()
errors = parser.get_errors()
```

## Running Tests with Different Scenarios

### Test Data Query Path
```bash
pytest tests/test_e2e_streaming.py::TestDataQueryE2E -v
```

### Test Error Handling
```bash
pytest tests/test_e2e_streaming.py::TestErrorHandlingE2E -v
```

### Test Protocol Compliance
```bash
pytest tests/test_e2e_streaming.py::TestDataStreamProtocol -v
```

### Run All with Coverage
```bash
pytest tests/test_e2e_streaming.py --cov=src --cov-report=html
```

## Mock Configuration Examples

### Setting Up a Data Query Test

```python
async def test_example(self, app_with_mocks):
    app_instance, router_structured, synthesis_structured = app_with_mocks

    # Configure intent classification
    router_structured.ainvoke = AsyncMock(return_value=IntentResult(
        intent="data_query",
        query_plan=[
            QueryPlanStep(
                tool_name="execute_read_query",
                mcp_server="data-mcp",
                parameters={"query": "SELECT * FROM ..."},
                description="Fetch data",
            ),
        ],
        reasoning="User is asking for data.",
    ))

    # Configure synthesis response
    synthesis_structured.ainvoke = AsyncMock(return_value=AnalyticsResponse(
        narrative="Results show...",
        components=[...],
    ))

    # Make request and verify response
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance)) as client:
        response = await client.post("/api/v1/analytics/chat", json={...})
```

## Key Testing Patterns

### 1. HTTPx AsyncClient with ASGI Transport
```python
async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_instance)) as client:
    response = await client.post("/api/v1/analytics/chat", json={...})
```

### 2. Parsing Data Stream Protocol
```python
parser = DataStreamParser()
parser.parse_response(response.text)
reasoning = parser.get_reasoning_tokens()
narrative = parser.get_text_tokens()
```

### 3. Verifying Event Sequences
```python
events = parser.events
# Find event indices and verify ordering
```

### 4. Mocking LLMs with Structured Output
```python
llm, structured = mock_router_llm
structured.ainvoke = AsyncMock(return_value=IntentResult(...))
```

## Troubleshooting

### Tests Timeout
- Increase `mock_config.mcp_startup_timeout`
- Check AsyncMock configuration
- Verify event generator isn't hung

### Protocol Parsing Failures
- Check JSON validity in response
- Verify line endings are correct
- Use DataStreamParser.parse_response() for debugging

### Mock Bridges Not Responding
- Verify `bridge.is_connected = True`
- Check `bridge.invoke()` returns proper dict
- Use `AsyncMock()` for async methods

## Coverage Goals

The test suite aims for:
- **Happy path**: All three intent paths (data_query, follow_up, clarification)
- **Error paths**: Tool failures, LLM failures, edge cases
- **Protocol compliance**: Format, headers, finish events, ordering
- **Multi-turn scenarios**: Conversation history, follow-ups

Total: **12 test methods** across **5 test classes**

## Integration with CI/CD

Add to your CI pipeline:

```yaml
# .github/workflows/test.yml
- name: Run E2E Streaming Tests
  run: |
    pytest tests/test_e2e_streaming.py -v \
      --tb=short \
      --junitxml=test-results.xml
```

## Future Enhancements

- [ ] Load testing with concurrent requests
- [ ] Stress test MCP bridge failures
- [ ] Test with real LLM models (not mocks)
- [ ] Benchmark streaming latency
- [ ] Test long-running streams (timeouts)
- [ ] Memory profiling for streaming responses
