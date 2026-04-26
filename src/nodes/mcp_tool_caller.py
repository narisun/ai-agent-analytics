"""
Analytics Agent — MCP Tool Caller Node.

Executes the query plan produced by the intent router by calling MCP tools
via MCPToolBridge instances. Supports parallel execution of independent
queries across different MCP servers.

Each tool call result is stored in raw_data_context keyed by a combination
of mcp_server and tool_name for uniqueness across multi-step plans.

Client name resolution:
  When a tool returns "client_not_found" or empty data for a client_name lookup,
  the node automatically queries the database for similar names using trigram
  or ILIKE matching. If close matches are found, the error message includes
  suggestions so the synthesis node can present them to the user.
"""
import asyncio
import json
import time

from platform_sdk import get_logger
from ..state import AnalyticsState

log = get_logger(__name__)


def _extract_client_name(step: dict) -> str | None:
    """Extract the client_name parameter from a plan step if present."""
    params = step.get("parameters", {})
    return params.get("client_name")


def _is_client_not_found(result) -> bool:
    """Check if a tool result indicates the client was not found."""
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                error = parsed.get("error", "")
                message = parsed.get("message", "")
                if error == "client_not_found":
                    return True
                if "not found" in str(message).lower():
                    return True
                if "no data" in str(message).lower():
                    return True
        except (json.JSONDecodeError, TypeError):
            pass
        # Check raw string
        lower = result.lower()
        if "not found" in lower or "no data" in lower or "no results" in lower:
            return True
    elif isinstance(result, dict):
        if result.get("error") == "client_not_found":
            return True
        if "not found" in str(result.get("message", "")).lower():
            return True
    return False


class MCPToolCallerNode:
    """Callable class for the MCP tool caller node.

    Constructor-injected with a ``tools_provider`` (MCPToolsProvider Protocol)
    that provides a flat list of all available langchain tools.

    See ``make_mcp_tool_caller_node`` for the backward-compat shim that wraps
    the legacy bridges-dict interface.
    """

    def __init__(self, *, tools_provider) -> None:
        self._tools_provider = tools_provider

    async def __call__(self, state: AnalyticsState, config=None) -> dict:
        plan = state.get("query_plan", [])
        session_id = state.get("session_id", "")

        # Extract user_ctx from LangGraph config if available
        user_ctx = (config or {}).get("configurable", {}).get("user_ctx")

        if not plan:
            log.warning("mcp_tool_caller_empty_plan")
            return {
                "raw_data_context": {},
                "active_tools": [],
                "errors": ["mcp_tool_caller: empty query plan"],
            }

        # Fetch all available tools once (shared across all steps)
        try:
            all_tools = await self._tools_provider.get_langchain_tools(user_ctx)
        except Exception as exc:
            log.error("mcp_tools_provider_error", error=str(exc))
            all_tools = []

        active_tools = []
        results = {}
        errors = []

        async def execute_step(step: dict, index: int) -> dict:
            """Execute a single plan step against the appropriate MCP server."""
            server_name = step.get("mcp_server", "unknown")
            tool_name = step.get("tool_name", "unknown")
            params = step.get("parameters", {})
            result_key = f"{server_name}:{tool_name}:{index}"

            tool_info = {
                "tool": tool_name,
                "server": server_name,
                "status": "running",
                "description": step.get("description", ""),
                "started_at": time.time(),
            }
            active_tools.append(tool_info)

            tool_fn = next((t for t in all_tools if t.name == tool_name), None)

            if tool_fn is None:
                available = [t.name for t in all_tools]
                error_msg = f"Tool '{tool_name}' not found on {server_name}. Available: {available}"
                log.error("mcp_tool_not_found", tool=tool_name, server=server_name, available=available)
                tool_info["status"] = "error"
                return {result_key: {"error": error_msg}}

            try:
                # Inject session_id if the tool accepts it.
                # Server-side injected — the LLM cannot override it.
                # User auth context (auth_context) is injected automatically
                # by the MCPToolBridge via ContextVar — not as a schema parameter.
                if tool_fn.args_schema:
                    tool_schema = tool_fn.args_schema.model_json_schema()
                    if "session_id" in tool_schema.get("properties", {}):
                        params["session_id"] = session_id or "analytics-default"

                result = await tool_fn.ainvoke(params)
                tool_info["status"] = "complete"
                tool_info["completed_at"] = time.time()
                log.info("mcp_tool_complete", tool=tool_name, server=server_name)
                return {result_key: result}

            except Exception as exc:
                tool_info["status"] = "error"
                tool_info["error"] = str(exc)
                log.error("mcp_tool_error", tool=tool_name, server=server_name, error=str(exc))
                return {result_key: {"error": str(exc)}}

        # Execute all plan steps concurrently
        tasks = [execute_step(step, i) for i, step in enumerate(plan)]
        step_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in step_results:
            if isinstance(result, Exception):
                errors.append(f"mcp_tool_caller: {result}")
            elif isinstance(result, dict):
                # Collect errors inline while merging results
                for key, value in result.items():
                    results[key] = value
                    if isinstance(value, dict) and "error" in value:
                        errors.append(f"{key}: {value['error']}")

        # Search for client name suggestions if any lookups failed
        client_names_tried = set()
        for step in plan:
            cn = _extract_client_name(step)
            if cn:
                client_names_tried.add(cn)

        if client_names_tried:
            has_not_found = False
            for key, value in results.items():
                if _is_client_not_found(value):
                    has_not_found = True
                    break

            if has_not_found:
                for client_name in client_names_tried:
                    similar = await self._find_similar_clients(
                        client_name, session_id, all_tools
                    )
                    if similar:
                        suggestion_msg = (
                            f"Client '{client_name}' was not found. "
                            f"Did you mean one of these? {', '.join(similar)}"
                        )
                        results["_client_suggestions"] = {
                            "searched_name": client_name,
                            "suggestions": similar,
                            "message": suggestion_msg,
                        }
                        log.info("client_suggestions_found",
                                 searched=client_name, suggestions=similar)
                    else:
                        results["_client_suggestions"] = {
                            "searched_name": client_name,
                            "suggestions": [],
                            "message": (
                                f"Client '{client_name}' was not found and no similar "
                                f"names were found in the database."
                            ),
                        }

        log.info(
            "mcp_tool_caller_complete",
            steps=len(plan),
            results=len(results),
            errors=len(errors),
        )

        return {
            "raw_data_context": results,
            "active_tools": active_tools,
            "errors": errors if errors else [],
        }

    async def _find_similar_clients(
        self, client_name: str, session_id: str, available_tools: list
    ) -> list[str]:
        """Query the database for client names similar to the given name.

        Uses parameterized ILIKE matching against both Salesforce Account and
        payments dim_party tables. The client_name is passed as a bind parameter
        to prevent SQL injection.
        """
        query_tool = next(
            (t for t in available_tools if t.name == "execute_read_query"), None
        )
        if not query_tool:
            return []

        try:
            # SECURITY: Use parameterized query to prevent SQL injection.
            # The $1 placeholder is bound to the ILIKE pattern at execution time.
            sql = """
                SELECT DISTINCT name FROM (
                    SELECT "Name" AS name
                    FROM salesforce."Account"
                    WHERE "Name" ILIKE $1
                    UNION
                    SELECT "PartyName" AS name
                    FROM bankdw.dim_party
                    WHERE "PartyName" ILIKE $1
                ) matches
                ORDER BY name
                LIMIT 5
            """
            like_pattern = f"%{client_name}%"

            params = {"query": sql, "params": [like_pattern]}
            if query_tool.args_schema:
                schema = query_tool.args_schema.model_json_schema()
                if "session_id" in schema.get("properties", {}):
                    params["session_id"] = session_id or "analytics-default"

            result = await query_tool.ainvoke(params)

            # Parse the result to extract names
            if isinstance(result, str):
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, list):
                        return [row.get("name", "") for row in parsed if row.get("name")]
                    elif isinstance(parsed, dict) and "rows" in parsed:
                        return [row.get("name", "") for row in parsed["rows"] if row.get("name")]
                except json.JSONDecodeError:
                    pass

            return []
        except Exception as exc:
            log.warning("similar_client_search_error", error=str(exc))
            return []


def make_mcp_tool_caller_node(bridges: dict):
    """Build the MCP tool caller node. Backward-compat shim.

    Args:
        bridges: Dict mapping MCP server names to MCPToolBridge instances.
                 Example: {"data-mcp": bridge1, "salesforce-mcp": bridge2, ...}
    """

    class _BridgeToolsProvider:
        async def get_langchain_tools(self, user_ctx=None):
            tools: list = []
            for bridge in (bridges or {}).values():
                if getattr(bridge, "is_connected", False):
                    bridge_tools = (
                        await bridge.get_langchain_tools(user_ctx)
                        if user_ctx
                        else await bridge.get_langchain_tools()
                    )
                    tools.extend(bridge_tools)
            return tools

    return MCPToolCallerNode(tools_provider=_BridgeToolsProvider())
