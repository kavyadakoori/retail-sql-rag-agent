import logging
from pathlib import Path
from typing import Any, Callable

from src.agents.state import AgentState
from src.tools.documents import search_documents
from src.tools.sql import execute_sql
from src.tools.sql_safety import is_destructive_request

logger = logging.getLogger(__name__)

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "database" / "schema_reference.md"
)
SCHEMA_TEXT = (
    SCHEMA_PATH.read_text(encoding="utf-8") if SCHEMA_PATH.exists() else ""
)


def _completed(state: AgentState, route: str) -> list[str]:
    completed = list(state.get("completed_routes", []) or [])
    if route not in completed:
        completed.append(route)
    return completed


def run_sql_specialist(
    state: AgentState, call_llm: Callable[[str], str]
) -> dict[str, Any]:
    query = state.get("active_query") or state["question"]
    if is_destructive_request(state.get("question", query)):
        return {
            "evidence": {
                **state.get("evidence", {}),
                "sql": {
                    "status": "rejected",
                    "reason": "Only read-only SQL queries are allowed.",
                },
            },
            "completed_routes": _completed(state, "sql"),
        }
    sql_prompt = f"""
You are an expert MySQL analyst working with a retail database.

TASK:
Generate exactly ONE valid MySQL SELECT query that answers
the user's question using the provided database schema.

STRICT RULES:

1. Use ONLY the tables and columns explicitly listed in
   the database schema.

2. Never invent table names, column names, relationships, or business
   definitions. SQL aliases may label selected expressions but are not schema
   columns.

3. Before deciding a calculation is unsupported, map the requested
   measures and dimensions to the available columns and check all documented
   relationships. A measure or dimension may require joining through another
   table; do not require its column to appear in the same table as the other
   part of the calculation.

4. Build direct or multi-table JOINs only from documented relationships.
   Use the documented key columns even when their names differ from the
   wording of the question; do not require a separate, differently named
   identifier when an existing documented key supports the relationship.

5. If the requested calculation can be performed using the available columns
   and relationships, generate the SQL query.

6. If the requested calculation genuinely cannot be performed after checking
   the available columns and join paths, return:
   UNSUPPORTED: <brief explanation of missing information>

7. Do not invent missing data or assume unavailable
   business information.

8. For return-rate questions, distinguish an order-based rate from a
   returned-unit rate. An order-based rate can use distinct orders with at
   least one linked return divided by distinct orders sold, with the requested
   grouping reached through documented joins. Protect counts from
   multiplication when an order has multiple return records. The returns
   table has no returned-quantity column, so a returned-unit rate cannot be
   calculated from this schema. If the question's intended rate is ambiguous,
   use the order-based interpretation and make that interpretation clear in
   the SQL result alias; never present it as a returned-unit rate.

9. Use LIMIT for requests involving top-N results.

10. Do not generate INSERT, UPDATE, DELETE, DROP, ALTER,
   CREATE, REPLACE, TRUNCATE, or multiple SQL statements.

11. Do not use SELECT * unless explicitly necessary.

12. Use DISTINCT when counting unique entities where
    joins or multiple records could cause duplicate counts.

13. Apply appropriate date filters when the question
    specifies a time period.

14. Return either:
    - One valid MySQL SELECT query, OR
    - An UNSUPPORTED: explanation response.

OUTPUT FORMAT:
- Return SQL only when the calculation is supported.
- Do not include Markdown code fences.
- Do not include explanations alongside SQL.
- Do not return JSON.
- Do not return an error object.
- Never return a fabricated SQL query.

REVENUE DEFINITION:

Revenue =
SUM(
    units_sold * unit_price * (1 - discount_pct / 100)
)

Use this definition only when the question asks for revenue.
Do not assume NULL discount_pct means zero unless documented.

DATABASE SCHEMA:
{SCHEMA_TEXT}

QUESTION:
{query}
"""
    sql = call_llm(sql_prompt).strip()
    logger.info("sql_query_generated", extra={"generated_sql": sql})
    # Handle an explicitly unsupported question.
    if sql.upper().startswith("UNSUPPORTED:"):
        reason = sql[len("UNSUPPORTED:"):].strip()

        logger.warning(
            "sql_generation_unsupported",
            extra={"reason": reason}
        )

        return {
            "evidence": {
                **state.get("evidence", {}),
                "sql": {
                    "status": "unsupported",
                    "reason": reason,
                },
            },
            "completed_routes": _completed(state, "sql"),
        }

    # Ensure the model returned a SELECT query before execution.
    if not sql.upper().startswith("SELECT"):
        logger.error(
            "sql_generation_invalid",
            extra={"generated_response": sql}
        )

        return {
            "evidence": {
                **state.get("evidence", {}),
                "sql": {
                    "status": "failed",
                    "reason": (
                        "The SQL model did not return a valid SELECT query."
                    ),
                },
            },
            "completed_routes": _completed(state, "sql"),
        }

    # Execute only after the response passes the basic format check.
    rows = execute_sql.invoke(sql)
    sql_evidence = {"status": "success", "query": sql, "rows": rows}
    return {
        "evidence": {**state.get("evidence", {}), "sql": sql_evidence},
        "completed_routes": _completed(state, "sql"),
    }


def run_document_specialist(state: AgentState) -> dict[str, Any]:
    query = state.get("active_query") or state["question"]
    context = search_documents.invoke(
        {
            "query": query,
            "tenant_id": state.get("tenant_id", "default"),
            "user_id": state.get("user_id", "anonymous"),
        }
    )
    return {
        "evidence": {**state.get("evidence", {}), "documents": context},
        "completed_routes": _completed(state, "documents"),
    }
