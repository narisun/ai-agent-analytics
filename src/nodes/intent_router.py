"""
Analytics Agent — Intent Router Node.

Classifies user intent and produces an MCP query execution plan.
Uses fast-routing (GPT-4o-mini) for low-latency classification.

Intent types:
  - data_query:    User wants to retrieve/analyze enterprise data → run MCP tools
  - follow_up:     User asks about previously retrieved data → skip tools, go to synthesis
  - clarification: Ambiguous request → ask user for more detail
"""
from typing import Callable, Optional

from langchain_core.messages import SystemMessage

from platform_sdk import get_logger
from platform_sdk.prompts import PromptLoader
from ..schemas.intent import IntentResult
from ..state import AnalyticsState

log = get_logger(__name__)


_ROUTER_SYSTEM_PROMPT_HEADER = """You are an intent classifier for an enterprise analytics platform.

Your job is to:
1. Classify the user's message into one of three intents.
2. If it is a data_query, produce a query_plan with the exact tool names and ALL required parameters.

## Available MCP Servers and Tools

{tool_catalog}

## Classification Rules (follow strictly)

**DEFAULT to data_query.** Most user messages are data queries. Only use the other intents
when you are highly confident they apply.

- **data_query** (MOST COMMON): The user mentions ANY data topic — revenue, payments, pipeline,
  clients, companies, accounts, trends, metrics, KPIs, news, or asks to "show", "get", "find",
  "what is", "how much", "compare", etc. Produce a query_plan.
  CRITICAL: Use ONLY exact tool names from the list above. Fill in ALL required parameters.
  If a required parameter like client_name is not explicit, infer it from context or use a
  reasonable default.

- **follow_up**: ONLY when the user explicitly refers to data already shown in this conversation
  (e.g. "break that down", "show as pie chart", "what about EMEA region").
  Do NOT produce a query_plan.

- **clarification**: ONLY when the message is truly unintelligible or has zero actionable content
  (e.g. "help", "hi", single letter). If you can make any reasonable interpretation, use data_query instead.

## Database Schema Reference (for SQL queries via data-mcp)

The PostgreSQL database has two schemas:

**salesforce schema** — CRM data. Tables and columns use Pascal case and MUST be double-quoted.
  ONLY these tables and columns exist — do NOT invent columns that are not listed here:

  salesforce."Account" ("Id", "Name", "Type", "Industry", "AccountNumber", "Ownership", "Phone", "Website", "BillingStreet", "BillingCity", "BillingState", "BillingPostalCode", "BillingCountry", "AnnualRevenue", "NumberOfEmployees", "Rating")
  salesforce."Opportunity" ("Id", "AccountId", "Pricebook2Id", "CampaignId", "Name", "StageName", "Amount", "CloseDate", "Type", "LeadSource", "Probability", "ForecastCategoryName", "NextStep", "Description")
  salesforce."Contact" ("Id", "AccountId", "FirstName", "LastName", "Email", "Phone", "Title", "Department", "MailingCity", "MailingState", "MailingCountry", "LeadSource")
  salesforce."Case" ("Id", "AccountId", "ContactId", "CaseNumber", "Subject", "Description", "Status", "Priority", "Origin", "Type", "Reason", "SuppliedEmail", "SuppliedPhone", "CreatedDate", "ClosedDate")
  salesforce."Lead" ("Id", "FirstName", "LastName", "Company", "Title", "Email", "Phone", "City", "State", "Country", "Status", "LeadSource", "Industry", "AnnualRevenue", "NumberOfEmployees", "Rating")
  salesforce."Contract" ("Id", "AccountId", "ContractNumber", "StartDate", "EndDate", "Status", "ContractTerm", "OwnerExpirationNotice", "SpecialTerms")
  salesforce."Campaign" ("Id", "Name", "Type", "Status", "StartDate", "EndDate", "BudgetedCost", "ActualCost", "ExpectedRevenue", "IsActive")
  salesforce."Task" ("Id", "Subject", "ActivityDate", "Status", "Priority", "WhoId", "WhatId", "Type", "Description")
  salesforce."Event" ("Id", "Subject", "StartDateTime", "EndDateTime", "IsAllDayEvent", "Location", "WhoId", "WhatId", "Type", "Description")
  salesforce."Product2" ("Id", "Name", "ProductCode", "Family", "IsActive", "Description")
  salesforce."OpportunityLineItem" ("Id", "OpportunityId", "PricebookEntryId", "Quantity", "UnitPrice", "TotalPrice", "ServiceDate", "Description")
  salesforce."OpportunityContactRole" ("Id", "OpportunityId", "ContactId", "Role", "IsPrimary")
  salesforce."CampaignMember" ("Id", "CampaignId", "ContactId", "LeadId", "Status", "HasResponded")
  salesforce."Pricebook2" ("Id", "Name", "IsActive", "Description")
  salesforce."PricebookEntry" ("Id", "Pricebook2Id", "Product2Id", "UnitPrice", "IsActive")

  NOTE: There is NO "Region" column. For geographic analysis, use "BillingState"/"BillingCountry" on Account,
  or JOIN Opportunity to Account via "AccountId" to get location data.

**payments schema** — Banking/payments data. Table names are lowercase (no quoting needed), but column names are Pascal case and MUST be double-quoted:

  payments.dim_party ("PartyKey", "PartyID", "PartyName", "PartyRoleType", "PartyType", "CustomerSegment", "KYCStatus", "RiskRating", "AMLRiskCategory", "SanctionsScreeningStatus", "PEPFlag", "IndustrySector", "CountryCode", "StateProvinceCode", "City", "PostalCode", "PreferredChannel", "FraudMonitoringSegment", "CustomerStatus", "OnboardingDate", "RelationshipStartDate")
  payments.dim_bank ("BankKey", "BankID", "BankName", "BankRoleType", "RoutingNumber", "SWIFTBIC", "BankType", "OwnershipType", "CountryCode", "HeadquartersState", "HeadquartersCity", "Regulator", "ClearingNetworksSupported", "CorrespondentBankFlag", "SettlementCurrency", "LiquidityTier", "BSAAMLProgramRating", "SanctionsComplianceStatus", "BankStatus")
  payments.dim_product ("ProductKey", "ProductID", "ProductName", "ProductCategory", "ProductFamily", "PaymentRail", "Directionality", "SettlementMethod", "SettlementSpeed", "TypicalUseCase", "RiskLevel", "CrossBorderCapability", "DefaultCurrency", "GeographyScope", "ProductStatus")
  payments.bridge_party_account ("PartyAccountKey", "PartyName", "AccountNumber", "BankName", "RoutingNumber", "PartyID", "BankID", "AccountType", "AccountStatus", "CurrencyCode")

SQL RULES:
  1. Always double-quote table names in salesforce schema: salesforce."Opportunity"
  2. Always double-quote ALL column names: "Name", "Amount", "StageName"
  3. ONLY use columns listed above. NEVER invent columns like "Region", "Revenue", "Quarter".

## EXAMPLES — query_plan with parameters (FOLLOW THIS FORMAT)

User: "Show me the payment trends for Acme Corp"
→ intent: data_query
→ query_plan:
  - tool_name: "get_payment_summary", mcp_server: "payments-mcp", parameters: {{"client_name": "Acme Corp"}}

User: "What's in the Salesforce pipeline for Microsoft?"
→ intent: data_query
→ query_plan:
  - tool_name: "get_salesforce_summary", mcp_server: "salesforce-mcp", parameters: {{"client_name": "Microsoft Corp."}}

User: "Show total opportunity amounts by stage"
→ intent: data_query
→ query_plan:
  - tool_name: "execute_read_query", mcp_server: "data-mcp", parameters: {{"query": "SELECT \"StageName\", COUNT(*) AS deal_count, SUM(\"Amount\") AS total_amount FROM salesforce.\"Opportunity\" GROUP BY \"StageName\" ORDER BY total_amount DESC"}}

User: "Show revenue by region"
→ intent: data_query
→ query_plan:
  - tool_name: "execute_read_query", mcp_server: "data-mcp", parameters: {{"query": "SELECT a.\"BillingState\", COUNT(*) AS deals, SUM(o.\"Amount\") AS total FROM salesforce.\"Opportunity\" o JOIN salesforce.\"Account\" a ON o.\"AccountId\" = a.\"Id\" GROUP BY a.\"BillingState\" ORDER BY total DESC"}}

User: "List all accounts by annual revenue"
→ intent: data_query
→ query_plan:
  - tool_name: "execute_read_query", mcp_server: "data-mcp", parameters: {{"query": "SELECT \"Name\", \"Industry\", \"AnnualRevenue\", \"Rating\" FROM salesforce.\"Account\" ORDER BY \"AnnualRevenue\" DESC"}}

User: "Show me payment products"
→ intent: data_query
→ query_plan:
  - tool_name: "execute_read_query", mcp_server: "data-mcp", parameters: {{"query": "SELECT \"ProductName\", \"ProductCategory\", \"PaymentRail\", \"RiskLevel\" FROM payments.dim_product WHERE \"ProductStatus\" = 'Active'"}}

User: "Show me recent news about Tesla"
→ intent: data_query
→ query_plan:
  - tool_name: "search_company_news", mcp_server: "news-search-mcp", parameters: {{"company_name": "Tesla"}}

IMPORTANT: The "parameters" field must contain the actual key-value pairs for the tool call.
Never leave parameters empty when the tool has required fields.
"""


async def _build_tool_catalog(bridges: dict) -> str:
    """Dynamically build a tool catalog from connected MCP bridges."""
    sections = []
    for server_name, bridge in bridges.items():
        if not bridge.is_connected:
            sections.append(f"### {server_name}\n(not connected)")
            continue
        try:
            tools = await bridge.get_langchain_tools()
            lines = [f"### {server_name}"]
            for tool in tools:
                # Include name, description, and required parameters
                schema = tool.args_schema.model_json_schema() if tool.args_schema else {}
                required = schema.get("required", [])
                props = schema.get("properties", {})
                # Filter out session_id — it is auto-injected by the tool caller
                AUTO_INJECTED = {"session_id"}
                param_parts = []
                for pname, pinfo in props.items():
                    if pname in AUTO_INJECTED:
                        continue
                    ptype = pinfo.get("type", "string")
                    pdesc = pinfo.get("description", "")
                    req_marker = " (REQUIRED)" if pname in required else " (optional)"
                    param_parts.append(f"    - {pname} ({ptype}){req_marker}: {pdesc}")
                params_str = "\n".join(param_parts) if param_parts else "    (no parameters)"
                lines.append(f"- **{tool.name}**: {tool.description}\n  Parameters:\n{params_str}")
            sections.append("\n".join(lines))
        except Exception as exc:
            log.warning("tool_catalog_error", server=server_name, error=str(exc))
            sections.append(f"### {server_name}\n(error loading tools: {exc})")
    return "\n\n".join(sections)


def make_intent_router_node(router_llm, bridges: dict, prompts: PromptLoader = None,
                            compaction_modifier: Optional[Callable] = None):
    """Build the intent router node.

    Args:
        router_llm: LangChain ChatModel configured with fast-routing.
        bridges: Dict mapping MCP server names to MCPToolBridge instances.
        prompts: Optional PromptLoader for template overrides.
        compaction_modifier: Optional callable to trim messages before LLM call.
    """
    structured_llm = router_llm.with_structured_output(IntentResult)
    _cached_prompt: list[str | None] = [None]  # mutable container for caching

    async def intent_router(state: AnalyticsState) -> dict:
        turn_count = state.get("turn_count", 0)
        has_data = bool(state.get("raw_data_context"))

        # Build system prompt with dynamic tool catalog (cached after first call)
        if _cached_prompt[0] is None:
            tool_catalog = await _build_tool_catalog(bridges)
            _cached_prompt[0] = _ROUTER_SYSTEM_PROMPT_HEADER.format(tool_catalog=tool_catalog)
            log.info("tool_catalog_built", length=len(tool_catalog))
        system_content = _cached_prompt[0]
        if has_data:
            available_keys = list((state.get("raw_data_context") or {}).keys())
            system_content += (
                f"\n\n## Session Context\n"
                f"Data already retrieved in this session: {available_keys}\n"
                f"Turn count: {turn_count}\n"
                f"If the user's question can be answered from existing data, classify as follow_up."
            )

        # Apply compaction to keep messages within token budget
        messages = list(state["messages"])
        if compaction_modifier is not None:
            messages = compaction_modifier({"messages": messages})

        try:
            result = await structured_llm.ainvoke(
                [SystemMessage(content=system_content)] + messages
            )

            plan_dump = [s.model_dump() for s in result.query_plan]
            log.info(
                "intent_classified",
                intent=result.intent,
                reasoning=result.reasoning,
                plan_steps=len(result.query_plan),
                plan=plan_dump,
                turn=turn_count + 1,
            )
            return {
                "intent": result.intent,
                "query_plan": plan_dump,
                "intent_reasoning": result.reasoning,
                "turn_count": turn_count + 1,
            }
        except Exception as exc:
            log.error("intent_router_error", error=str(exc))
            # Fallback: if we have data, treat as follow_up; otherwise clarify
            if has_data:
                return {
                    "intent": "follow_up",
                    "query_plan": [],
                    "intent_reasoning": f"Router error — falling back to follow_up: {exc}",
                    "turn_count": turn_count + 1,
                    "errors": [f"intent_router: {exc}"],
                }
            return {
                "intent": "clarification",
                "query_plan": [],
                "intent_reasoning": f"Router error — requesting clarification: {exc}",
                "turn_count": turn_count + 1,
                "errors": [f"intent_router: {exc}"],
            }

    return intent_router


def route_after_intent(state: AnalyticsState) -> str:
    """Conditional edge: dispatch based on classified intent."""
    intent = state.get("intent", "data_query")
    if intent == "data_query":
        return "mcp_tool_caller"
    elif intent == "follow_up":
        return "synthesis"
    else:
        return "error_handler"  # clarification → send message back to user
