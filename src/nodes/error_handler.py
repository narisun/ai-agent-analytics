"""
Analytics Agent — Error Handler Node.

Handles clarification requests and general error fallback.
Produces a user-facing message without requiring MCP tool calls.
"""
from langchain_core.messages import AIMessage

from platform_sdk import get_logger
from ..state import AnalyticsState

log = get_logger(__name__)


def make_error_handler_node():
    """Build the error handler node."""

    async def error_handler(state: AnalyticsState) -> dict:
        intent = state.get("intent", "clarification")
        reasoning = state.get("intent_reasoning", "")
        errors = state.get("errors", [])

        if intent == "clarification":
            message = (
                "I'd like to help you analyze your data, but I need a bit more detail. "
                "Could you specify:\n\n"
                "- **What data** you're looking for (revenue, pipeline, payments, news)\n"
                "- **What entity** (company name, region, time period)\n"
                "- **What format** you'd like (chart, table, summary)\n\n"
                "For example: *\"Show me Q3 revenue by region\"* or "
                "*\"What's the payment trend for Acme Corp?\"*"
            )
            log.info("clarification_sent", reasoning=reasoning)
        else:
            error_detail = "; ".join(errors) if errors else "Unknown error"
            message = (
                f"I encountered an issue processing your request: {error_detail}. "
                "Could you try rephrasing your question?"
            )
            log.warning("error_handler_invoked", errors=errors)

        return {
            "narrative": message,
            "ui_components": [],
            "messages": [AIMessage(content=message)],
        }

    return error_handler
