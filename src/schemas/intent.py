"""
Analytics Agent — Intent classification schemas.

Used by the intent_router node to produce structured classification output.
The query_plan guides the MCP tool caller on which tools to invoke.
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class QueryPlanStep(BaseModel):
    """A single step in the MCP query execution plan."""
    tool_name: str = Field(
        description="Exact MCP tool name from the catalog (e.g. 'execute_read_query', 'get_salesforce_summary')",
    )
    mcp_server: str = Field(
        description="Target MCP server: 'data-mcp' | 'salesforce-mcp' | 'payments-mcp' | 'news-search-mcp'",
    )
    parameters: dict = Field(
        default_factory=dict,
        description=(
            "Tool invocation parameters as key-value pairs. "
            "MUST include all required parameters for the tool. "
            'Examples: {"query": "SELECT ..."} for execute_read_query, '
            '{"client_name": "Acme Corp"} for get_payment_summary.'
        ),
    )
    description: str = Field(
        default="",
        description="Human-readable description of what this step does (shown in trace panel)",
    )


class IntentResult(BaseModel):
    """Structured output from the intent router node."""
    intent: Literal["data_query", "follow_up", "clarification"] = Field(
        description=(
            "data_query: user wants to retrieve or analyze data. "
            "follow_up: user is asking about previously retrieved data. "
            "clarification: user's request is ambiguous and needs more info."
        ),
    )
    query_plan: list[QueryPlanStep] = Field(
        default_factory=list,
        description="Execution plan for data_query intents. Empty for follow_up/clarification.",
    )
    reasoning: str = Field(
        default="",
        description="Chain-of-thought reasoning explaining the classification decision",
    )
    clarification_message: Optional[str] = Field(
        default=None,
        description="Message to send to user when intent is 'clarification'",
    )
