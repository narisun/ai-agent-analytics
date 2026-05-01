"""Shared pytest fixtures for end-to-end and integration tests.

Provides:
  - mock_bridges: Dict of mock MCP bridges with realistic tool schemas
  - mock_config: AgentConfig-like mock with model routing
  - mock_router_llm_factory: Factory for router LLM mocks
  - mock_synthesis_llm_factory: Factory for synthesis LLM mocks
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from platform_sdk import AgentConfig


@pytest.fixture
def mock_bridges():
    """Create a dict of mock MCP bridges with realistic tool schemas.

    Each bridge:
      - is_connected = True
      - get_langchain_tools() returns a list of mock tools with args_schema
      - invoke() can be called with tool_name and parameters
    """
    bridges = {}

    # Helper to create a mock tool with schema
    def make_mock_tool(
        name: str, description: str, required_params: list[str], all_params: dict[str, str]
    ):
        """Create a mock LangChain tool with args_schema."""
        tool = MagicMock()
        tool.name = name
        tool.description = description

        # Mock the args_schema (Pydantic model schema)
        schema = MagicMock()
        schema.model_json_schema.return_value = {
            "type": "object",
            "properties": all_params,
            "required": required_params,
        }
        tool.args_schema = schema
        return tool

    # data-mcp: Database query tool
    data_mcp = MagicMock()
    data_mcp.is_connected = True
    data_mcp.get_langchain_tools = AsyncMock(
        return_value=[
            make_mock_tool(
                "execute_read_query",
                "Execute a read-only SQL query against the enterprise data warehouse",
                ["query"],
                {"query": "string"},
            ),
        ]
    )
    data_mcp.invoke = AsyncMock(
        side_effect=lambda tool_name, params: {
            "status": "success",
            "rows": [
                {"category": "North America", "value": 1200000},
                {"category": "EMEA", "value": 800000},
                {"category": "APAC", "value": 600000},
            ],
            "row_count": 3,
            "execution_time_ms": 150,
        }
    )
    bridges["data-mcp"] = data_mcp

    # salesforce-mcp: Salesforce-specific tools
    salesforce_mcp = MagicMock()
    salesforce_mcp.is_connected = True
    salesforce_mcp.get_langchain_tools = AsyncMock(
        return_value=[
            make_mock_tool(
                "get_salesforce_summary",
                "Retrieve a summary of Salesforce data for an account (pipeline, contacts, activities)",
                ["client_name"],
                {"client_name": "string"},
            ),
            make_mock_tool(
                "get_account_details",
                "Get detailed information about a Salesforce account",
                ["account_id"],
                {"account_id": "string"},
            ),
        ]
    )
    salesforce_mcp.invoke = AsyncMock(
        side_effect=lambda tool_name, params: {
            "status": "success",
            "account": {
                "id": "001xx000003DHyAAM",
                "name": "Acme Corp",
                "industry": "Manufacturing",
                "annual_revenue": 5000000,
            },
            "pipeline": {
                "total_value": 2500000,
                "opportunity_count": 15,
                "average_deal_size": 166667,
            },
        }
    )
    bridges["salesforce-mcp"] = salesforce_mcp

    # payments-mcp: Payment and transaction tools
    payments_mcp = MagicMock()
    payments_mcp.is_connected = True
    payments_mcp.get_langchain_tools = AsyncMock(
        return_value=[
            make_mock_tool(
                "get_payment_summary",
                "Get payment and transaction summary for a client",
                ["client_name"],
                {"client_name": "string"},
            ),
            make_mock_tool(
                "get_settlement_status",
                "Get current settlement status and pending transactions",
                ["client_id"],
                {"client_id": "string"},
            ),
        ]
    )
    payments_mcp.invoke = AsyncMock(
        side_effect=lambda tool_name, params: {
            "status": "success",
            "client": params.get("client_name", "Unknown"),
            "total_transactions": 1250,
            "total_volume": 47500000,
            "settlement_status": "settled",
            "pending_transactions": 3,
        }
    )
    bridges["payments-mcp"] = payments_mcp

    # news-search-mcp: News and external data
    news_mcp = MagicMock()
    news_mcp.is_connected = True
    news_mcp.get_langchain_tools = AsyncMock(
        return_value=[
            make_mock_tool(
                "search_company_news",
                "Search for recent news articles about a company",
                ["company_name"],
                {"company_name": "string"},
            ),
        ]
    )
    news_mcp.invoke = AsyncMock(
        side_effect=lambda tool_name, params: {
            "status": "success",
            "company": params.get("company_name", "Unknown"),
            "articles": [
                {
                    "title": "Company announces new product",
                    "source": "Bloomberg",
                    "date": "2024-04-01",
                    "sentiment": "positive",
                },
                {
                    "title": "Quarterly earnings exceed expectations",
                    "source": "Reuters",
                    "date": "2024-03-28",
                    "sentiment": "positive",
                },
            ],
            "article_count": 2,
        }
    )
    bridges["news-search-mcp"] = news_mcp

    return bridges


@pytest.fixture
def mock_config():
    """Create an AgentConfig-like mock with model routing.

    Returns a MagicMock that has:
      - router_model_route: "gpt-4o-mini"
      - synthesis_model_route: "gpt-4o"
      - mcp_startup_timeout: 30
    """
    config = MagicMock(spec=AgentConfig)
    config.router_model_route = "gpt-4o-mini"
    config.synthesis_model_route = "gpt-4o"
    config.mcp_startup_timeout = 30
    config.max_history_turns = 5
    config.max_parallel_tools = 4
    return config


@pytest.fixture
def mock_router_llm_factory():
    """Factory that returns a mock router LLM.

    The returned LLM can be configured with:
      .with_structured_output(IntentResult) → AsyncMock that returns IntentResult

    Returns:
      A factory function that, when called, returns (llm, structured_mock)
    """

    def factory():
        llm = MagicMock()
        structured = AsyncMock()
        llm.with_structured_output = MagicMock(return_value=structured)
        return llm, structured

    return factory


@pytest.fixture
def mock_synthesis_llm_factory():
    """Factory that returns a mock synthesis LLM.

    The returned LLM can be configured with:
      .with_structured_output(AnalyticsResponse) → AsyncMock that returns AnalyticsResponse

    Returns:
      A factory function that, when called, returns (llm, structured_mock)
    """

    def factory():
        llm = MagicMock()
        structured = AsyncMock()
        llm.with_structured_output = MagicMock(return_value=structured)
        return llm, structured

    return factory


@pytest.fixture
def mock_router_llm(mock_router_llm_factory):
    """Convenience fixture: returns a configured mock router LLM."""
    return mock_router_llm_factory()


@pytest.fixture
def mock_synthesis_llm(mock_synthesis_llm_factory):
    """Convenience fixture: returns a configured mock synthesis LLM."""
    return mock_synthesis_llm_factory()
