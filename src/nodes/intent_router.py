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

- **follow_up**: ONLY when the user explicitly refers to data already displayed and wants
  a different VIEW, FORMAT, or BREAKDOWN of that same data. Examples:
    - "break that down by region" (same data, different slice)
    - "show as a pie chart" (same data, different format)
    - "what percentage is wire transfers" (drill into existing results)
  NOT follow_up: "show me IBM data", "just get payments for X", "try again" — these are data_query.
  Do NOT produce a query_plan.

- **clarification**: Use when:
  (a) The user mentions a metric that maps to MULTIPLE schemas and the answer would differ
      depending on which is used. Examples:
      - "revenue" → could be salesforce."Account"."AnnualRevenue" (reported revenue),
        salesforce."Opportunity"."Amount" (pipeline value), OR bankdw.fact_payments."Amount"
        (actual payment volume). Ask which domain they mean.
      - "transactions" → could be salesforce."Task"/"Event" (activities) or
        bankdw.fact_payments (payment transactions). Ask which.
  (b) A query requires a specific client name but none was provided AND the question is NOT
      about aggregate/all-client analysis.
  (c) The message is truly unintelligible or has zero actionable content (e.g. "help", "hi").

  When in doubt between data_query and clarification:
    - If the query maps to ONE obvious table/tool → data_query
    - If it maps to 2+ tables and the answer would differ significantly → clarification
    - If you proceed with an assumption, STATE it clearly in the reasoning field
      (e.g. "Assuming user means Opportunity pipeline, not payment volume")

## Database Schema Reference (for SQL queries via data-mcp)

The PostgreSQL database has two schemas:

**salesforce schema** — CRM data. Tables and columns use Pascal case and MUST be double-quoted.
  ONLY these tables and columns exist — do NOT invent columns that are not listed here:

  salesforce."Account" ("Id", "Name", "Type", "Industry", "AccountNumber", "Ownership", "Phone", "Website", "BillingStreet", "BillingCity", "BillingState", "BillingPostalCode", "BillingCountry", "AnnualRevenue", "NumberOfEmployees", "Rating")
    — "Type" values: 'Customer - Direct', 'Customer - Channel', 'Technology Partner', 'Installation Partner'
    — "Industry" values: 'Banking Client', 'Technology', 'Healthcare', 'Energy', 'Manufacturing', 'Retail', 'Financial Services'
    — "Rating" values: 'Hot', 'Warm', 'Cold'

  salesforce."Opportunity" ("Id", "AccountId", "Pricebook2Id", "CampaignId", "Name", "StageName", "Amount", "CloseDate", "Type", "LeadSource", "Probability", "ForecastCategoryName", "NextStep", "Description")
    — "StageName" values: 'Prospecting', 'Qualification', 'Needs Analysis', 'Value Proposition', 'Id. Decision Makers', 'Perception Analysis', 'Proposal/Price Quote', 'Negotiation/Review', 'Closed Won', 'Closed Lost'
    — "ForecastCategoryName" values: 'Pipeline', 'Best Case', 'Commit', 'Omitted', 'Closed'
    — "Type" values: 'New Customer', 'Existing Customer - Upgrade', 'Existing Customer - Replacement', 'Existing Customer - Downgrade'
    — "Amount" is deal value in USD (measure|financial)

  salesforce."Contact" ("Id", "AccountId", "FirstName", "LastName", "Email", "Phone", "Title", "Department", "MailingCity", "MailingState", "MailingCountry", "LeadSource")
  salesforce."Case" ("Id", "AccountId", "ContactId", "CaseNumber", "Subject", "Description", "Status", "Priority", "Origin", "Type", "Reason", "SuppliedEmail", "SuppliedPhone", "CreatedDate", "ClosedDate")
    — "Status" values: 'New', 'Working', 'Escalated', 'Closed'
    — "Priority" values: 'High', 'Medium', 'Low'

  salesforce."Lead" ("Id", "FirstName", "LastName", "Company", "Title", "Email", "Phone", "City", "State", "Country", "Status", "LeadSource", "Industry", "AnnualRevenue", "NumberOfEmployees", "Rating")
    — "Status" values: 'Open - Not Contacted', 'Working - Contacted', 'Closed - Converted', 'Closed - Not Converted'

  salesforce."Contract" ("Id", "AccountId", "ContractNumber", "StartDate", "EndDate", "Status", "ContractTerm", "OwnerExpirationNotice", "SpecialTerms")
    — "Status" values: 'Draft', 'In Approval Process', 'Activated'

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

**bankdw schema** — Banking/payments data warehouse. Table names are lowercase, column names
  are Pascal case and MUST be double-quoted:

  bankdw.fact_payments ("TransactionID", "TransactionDate", "PayorName", "PayorAccountNumber", "PayorBank", "PayorRoutingNumber", "PayeeName", "PayeeAccountNumber", "PayeeBank", "PayeeRoutingNumber", "TransactionType", "Amount", "Currency", "Status")
    — Central fact table for all payment transactions
    — TRANSACTION MODEL: Every transaction has TWO parties:
        • "PayorName" = the SENDER (company sending money)
        • "PayeeName" = the RECEIVER (company receiving money)
      Both are company names that join to dim_party."PartyName" and salesforce."Account"."Name".
      IMPORTANT: When the user says "party", "counterparty", "client", or "company" in the context
      of a transaction, they may mean EITHER side. Always consider BOTH "PayorName" AND "PayeeName"
      unless the user specifically says "sender"/"payor" or "receiver"/"payee".
      Example: "which party of Costco uses wires most" → query BOTH sides:
        WHERE ("PayorName" ILIKE '%Costco%' OR "PayeeName" ILIKE '%Costco%')
      Then GROUP BY the OTHER party to show counterparties, or GROUP BY role to show direction.
    — "PayorBank"/"PayeeBank" = bank name, joins to dim_bank."BankName"
    — "TransactionType" values: 'ACH Credit', 'ACH Debit', 'Wire Transfer', 'RTP Credit', 'Check Payment', 'ACH Return', 'Wire Return'
    — "Status" values: 'Completed', 'Pending', 'Failed', 'Returned', 'Reversed'
    — "Amount" is transaction value in USD (measure|financial)
    — "Currency" is always 'USD'

  bankdw.dim_party ("PartyKey", "PartyID", "PartyName", "PartyRoleType", "PartyType", "CustomerSegment", "KYCStatus", "RiskRating", "AMLRiskCategory", "SanctionsScreeningStatus", "PEPFlag", "IndustrySector", "CountryCode", "StateProvinceCode", "City", "PostalCode", "PreferredChannel", "FraudMonitoringSegment", "CustomerStatus", "OnboardingDate", "RelationshipStartDate")
    — "PartyName" is the key join column to fact_payments and salesforce."Account"."Name"
    — "CustomerSegment" values: 'Retail', 'Commercial', 'Corporate', 'Institutional'
    — "RiskRating" values: 'Low', 'Medium', 'Elevated', 'High'
    — "CustomerStatus" values: 'Active', 'Inactive', 'Suspended', 'Closed'

  bankdw.dim_bank ("BankKey", "BankID", "BankName", "BankRoleType", "RoutingNumber", "SWIFTBIC", "BankType", "OwnershipType", "CountryCode", "HeadquartersState", "HeadquartersCity", "Regulator", "ClearingNetworksSupported", "CorrespondentBankFlag", "SettlementCurrency", "LiquidityTier", "BSAAMLProgramRating", "SanctionsComplianceStatus", "BankStatus")
    — "BankName" is the key join column to fact_payments."PayorBank"/"PayeeBank"
    — "BankType" values: 'National Bank', 'State Bank', 'Credit Union', 'Savings Institution'

  bankdw.dim_product ("ProductKey", "ProductID", "ProductName", "ProductCategory", "ProductFamily", "PaymentRail", "Directionality", "SettlementMethod", "SettlementSpeed", "TypicalUseCase", "RiskLevel", "CrossBorderCapability", "DefaultCurrency", "GeographyScope", "ProductStatus")
    — "ProductCategory" values: 'Credit Transfer', 'Debit Transfer', 'Check', 'Real-Time'
    — "PaymentRail" values: 'ACH Credit', 'ACH Debit', 'Fedwire', 'RTP', 'Check Clearing'

  bankdw.bridge_party_account ("PartyAccountKey", "PartyName", "AccountNumber", "BankName", "RoutingNumber", "PartyID", "BankID", "AccountType", "AccountStatus", "CurrencyCode")
    — Links parties to their bank accounts; use to find which bank a party uses

## JOIN RELATIONSHIPS (use these exact column pairs)

### Within bankdw schema (text-based joins — values must match exactly):
  fact_payments."PayorName"  = dim_party."PartyName"     (text match — payor lookup)
  fact_payments."PayeeName"  = dim_party."PartyName"     (text match — payee lookup)
  fact_payments."PayorBank"  = dim_bank."BankName"       (text match — payor bank)
  fact_payments."PayeeBank"  = dim_bank."BankName"       (text match — payee bank)
  bridge_party_account."PartyID" = dim_party."PartyID"   (FK — party accounts)
  bridge_party_account."BankID"  = dim_bank."BankID"     (FK — account bank)

### Cross-schema (salesforce ↔ bankdw — the bridge between CRM and payments):
  salesforce."Account"."Name"  = bankdw.dim_party."PartyName"        (text match)
  salesforce."Account"."Name"  = bankdw.fact_payments."PayorName"    (text match)
  salesforce."Account"."Name"  = bankdw.fact_payments."PayeeName"    (text match)

### Within salesforce schema (FK-based joins):
  "Opportunity"."AccountId"          = "Account"."Id"
  "Contact"."AccountId"              = "Account"."Id"
  "Case"."AccountId"                 = "Account"."Id"
  "Contract"."AccountId"             = "Account"."Id"
  "OpportunityLineItem"."OpportunityId" = "Opportunity"."Id"
  "OpportunityContactRole"."OpportunityId" = "Opportunity"."Id"
  "CampaignMember"."CampaignId"     = "Campaign"."Id"

## TOOL ROUTING RULES (MANDATORY — follow these BEFORE writing any query_plan)

RULE 1 — CLIENT-SPECIFIC PAYMENT QUERIES → get_payment_summary:
  When the user asks about payments, transactions, or banking data for a SPECIFIC client/company,
  ALWAYS use get_payment_summary on payments-mcp. NEVER write raw SQL against bankdw tables
  with a client name filter. The tool handles fuzzy name matching (e.g. "Pepsi" → "PepsiCo Inc.").
  ✅ get_payment_summary with client_name="Pepsi"
  ❌ execute_read_query with WHERE "PayorName" = 'Pepsi'

RULE 2 — CLIENT-SPECIFIC CRM QUERIES → get_salesforce_summary:
  When the user asks about CRM data for a SPECIFIC client, ALWAYS use get_salesforce_summary
  on salesforce-mcp. It handles fuzzy name matching too.
  ✅ get_salesforce_summary with client_name="IBM"
  ❌ execute_read_query with WHERE "Name" = 'IBM'

RULE 3 — AGGREGATE/CROSS-CLIENT QUERIES → execute_read_query:
  Use raw SQL via data-mcp ONLY for aggregate analysis across ALL clients, or queries
  that don't filter by a specific client name (e.g. "total payments by type",
  "top 10 accounts by revenue", "pipeline by stage").

RULE 4 — COMBINED CLIENT VIEW → parallel get_salesforce_summary + get_payment_summary:
  When user wants both CRM and payment data for one client, use BOTH tools in parallel.

RULE 5 — MULTI-CLIENT / RELATIONSHIP QUERIES → execute_read_query with ILIKE:
  When the user asks about transactions BETWEEN two companies (e.g. "transactions between
  Google and Costco"), use execute_read_query since get_payment_summary only accepts one client.
  But ALWAYS use ILIKE for name matching, never exact = match:
  ✅ WHERE "PayorName" ILIKE '%Google%' AND "PayeeName" ILIKE '%Costco%'
  ❌ WHERE "PayorName" = 'Google' AND "PayeeName" = 'Costco'
  Also include the reverse direction with OR:
  ✅ WHERE ("PayorName" ILIKE '%Google%' AND "PayeeName" ILIKE '%Costco%')
        OR ("PayorName" ILIKE '%Costco%' AND "PayeeName" ILIKE '%Google%')

## SQL RULES (only for execute_read_query on data-mcp)
  1. Always double-quote table names in salesforce schema: salesforce."Opportunity"
  2. Always double-quote ALL column names: "Name", "Amount", "StageName"
  3. ONLY use columns listed above. NEVER invent columns like "Region", "Revenue", "Quarter".
  4. For bankdw tables use schema prefix: bankdw.fact_payments, bankdw.dim_party
  5. Always include LIMIT when exploring tables (default LIMIT 100).
  6. For ANY client/company name filter in raw SQL, ALWAYS use ILIKE with % wildcards:
     ✅ WHERE "PayorName" ILIKE '%Pepsi%'
     ❌ WHERE "PayorName" = 'Pepsi'
     This ensures partial names like "Pepsi" match "PepsiCo Inc." in the database.

## EXAMPLES — query_plan with parameters (FOLLOW THIS FORMAT)

User: "Show me the payment trends for Acme Corp"
→ intent: data_query
→ query_plan:
  - tool_name: "get_payment_summary", mcp_server: "payments-mcp", parameters: {{"client_name": "Acme Corp"}}
  (NOT execute_read_query — client-specific payment query must use get_payment_summary)

User: "Show me transactions for Pepsi"
→ intent: data_query
→ query_plan:
  - tool_name: "get_payment_summary", mcp_server: "payments-mcp", parameters: {{"client_name": "Pepsi"}}
  (Fuzzy matching will resolve "Pepsi" to the actual party name like "PepsiCo Inc.")

User: "What's in the Salesforce pipeline for Microsoft?"
→ intent: data_query
→ query_plan:
  - tool_name: "get_salesforce_summary", mcp_server: "salesforce-mcp", parameters: {{"client_name": "Microsoft"}}
  (Fuzzy matching will resolve "Microsoft" to the actual account name)

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
  - tool_name: "execute_read_query", mcp_server: "data-mcp", parameters: {{"query": "SELECT \"Name\", \"Industry\", \"AnnualRevenue\", \"Rating\" FROM salesforce.\"Account\" ORDER BY \"AnnualRevenue\" DESC LIMIT 50"}}

User: "Show me payment products"
→ intent: data_query
→ query_plan:
  - tool_name: "execute_read_query", mcp_server: "data-mcp", parameters: {{"query": "SELECT \"ProductName\", \"ProductCategory\", \"PaymentRail\", \"RiskLevel\" FROM bankdw.dim_product WHERE \"ProductStatus\" = 'Active'"}}

User: "Show me recent news about Tesla"
→ intent: data_query
→ query_plan:
  - tool_name: "search_company_news", mcp_server: "news-search-mcp", parameters: {{"company_name": "Tesla"}}

User: "Show payment volumes for our top accounts by opportunity size"
→ intent: data_query
→ query_plan:
  - tool_name: "execute_read_query", mcp_server: "data-mcp", parameters: {{"query": "SELECT a.\"Name\", SUM(o.\"Amount\") AS pipeline_value, SUM(fp.\"Amount\") AS payment_volume FROM salesforce.\"Account\" a JOIN salesforce.\"Opportunity\" o ON o.\"AccountId\" = a.\"Id\" JOIN bankdw.fact_payments fp ON fp.\"PayorName\" = a.\"Name\" WHERE o.\"StageName\" NOT IN ('Closed Lost') AND fp.\"Status\" = 'Completed' GROUP BY a.\"Name\" ORDER BY pipeline_value DESC LIMIT 10"}}

User: "Full view of Acme Corp — CRM and payments"
→ intent: data_query
→ query_plan (multi-step, parallel):
  - tool_name: "get_salesforce_summary", mcp_server: "salesforce-mcp", parameters: {{"client_name": "Acme Corp"}}
  - tool_name: "get_payment_summary", mcp_server: "payments-mcp", parameters: {{"client_name": "Acme Corp"}}

User: "What are the total completed payments by transaction type?"
→ intent: data_query
→ query_plan:
  - tool_name: "execute_read_query", mcp_server: "data-mcp", parameters: {{"query": "SELECT \"TransactionType\", COUNT(*) AS txn_count, SUM(\"Amount\") AS total_amount FROM bankdw.fact_payments WHERE \"Status\" = 'Completed' GROUP BY \"TransactionType\" ORDER BY total_amount DESC"}}

User: "Show me transactions between Google and Costco"
→ intent: data_query
→ query_plan:
  - tool_name: "execute_read_query", mcp_server: "data-mcp", parameters: {{"query": "SELECT \"TransactionDate\", \"PayorName\", \"PayeeName\", \"TransactionType\", \"Amount\", \"Status\" FROM bankdw.fact_payments WHERE (\"PayorName\" ILIKE '%Google%' AND \"PayeeName\" ILIKE '%Costco%') OR (\"PayorName\" ILIKE '%Costco%' AND \"PayeeName\" ILIKE '%Google%') ORDER BY \"TransactionDate\" DESC LIMIT 50"}}
  (Multi-client relationship query — uses ILIKE for fuzzy matching, checks both directions)

User: "Which party of Costco uses wires most?"
→ intent: data_query
→ query_plan:
  - tool_name: "execute_read_query", mcp_server: "data-mcp", parameters: {{"query": "SELECT CASE WHEN \"PayorName\" ILIKE '%Costco%' THEN \"PayeeName\" ELSE \"PayorName\" END AS counterparty, CASE WHEN \"PayorName\" ILIKE '%Costco%' THEN 'Costco sends to' ELSE 'Costco receives from' END AS direction, COUNT(*) AS wire_count, SUM(\"Amount\") AS total_amount FROM bankdw.fact_payments WHERE (\"PayorName\" ILIKE '%Costco%' OR \"PayeeName\" ILIKE '%Costco%') AND \"TransactionType\" = 'Wire Transfer' AND \"Status\" = 'Completed' GROUP BY counterparty, direction ORDER BY wire_count DESC LIMIT 10"}}
  ("party" means counterparty on EITHER side — check both PayorName and PayeeName)

User: "Show me revenue"
→ intent: clarification
→ reasoning: "Ambiguous — 'revenue' could mean Account.AnnualRevenue (reported company revenue), Opportunity.Amount (pipeline deal value), or fact_payments.Amount (actual payment volume). Need to ask which."

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
            raw_context = state.get("raw_data_context") or {}
            available_keys = list(raw_context.keys())
            # Detect if previous results had errors (no data, client not found, etc.)
            has_errors = any(
                (isinstance(v, dict) and "error" in v)
                or (isinstance(v, str) and ("no_data" in v or "client_not_found" in v or "not found" in v.lower()))
                for v in raw_context.values()
            )
            prior_errors = state.get("errors", [])
            error_note = ""
            if has_errors or prior_errors:
                error_note = (
                    f"\n⚠️ IMPORTANT: Previous queries returned errors or no data. "
                    f"If the user retries or rephrases, classify as data_query to re-fetch.\n"
                )
            system_content += (
                f"\n\n## Session Context\n"
                f"Data already retrieved in this session: {available_keys}\n"
                f"Turn count: {turn_count}\n"
                f"{error_note}\n"
                f"### follow_up vs data_query with existing data\n"
                f"Classify as follow_up ONLY when the user explicitly references the data "
                f"already shown (e.g. 'break that down by region', 'show as a chart', "
                f"'what percentage is wire transfers').\n\n"
                f"Classify as data_query when:\n"
                f"- The user mentions a NEW client/company name not in the existing data\n"
                f"- The user rephrases or retries a failed query (e.g. 'just show me IBM data')\n"
                f"- The user asks for a DIFFERENT data domain (e.g. payments vs CRM)\n"
                f"- The user says 'show', 'get', 'find' + an entity name\n"
                f"- The previous results contained errors or 'no data' responses\n\n"
                f"When in doubt, prefer data_query over follow_up — it's better to re-fetch "
                f"than to give the user an empty response."
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
