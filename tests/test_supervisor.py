import json
import sqlite3
from uuid import uuid4

import pytest
from types import SimpleNamespace
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agents import graph as agent_graph
from src.agents.supervisor import choose_route, finalize
from src.agents import specialists
from src.agents.graph import conversation_config


def test_supervisor_routes_empty_state_to_available_specialist():
    result = choose_route(
        {
            "question": "What is the return policy?",
            "evidence": {},
            "completed_routes": [],
            "pending_routes": [],
            "iteration_count": 0,
        },
        lambda prompt: '{"routes":["documents"]}',
    )
    assert result["route"] == "documents"


def test_supervisor_rejects_invalid_llm_route_shape():
    with pytest.raises(ValueError, match="did not return a valid"):
        choose_route(
            {
                "question": "What is the return policy?",
                "evidence": {},
                "completed_routes": [],
                "iteration_count": 0,
            },
            lambda prompt: '{"routes":"documents"}',
        )


def test_policy_duration_question_does_not_route_to_sql():
    result = choose_route(
        {
            "question": (
                "According to the return policy, how many days do I have "
                "to return an unopened item?"
            ),
            "evidence": {},
            "completed_routes": [],
            "iteration_count": 0,
        },
        lambda prompt: '{"routes":["documents"]}',
    )

    assert result["routes"] == ["documents"]
    assert result["route"] == "documents"


def test_standard_delivery_question_routes_to_documents_only():
    result = choose_route(
        {
            "question": "How long does standard delivery take?",
            "evidence": {},
            "completed_routes": [],
            "iteration_count": 0,
        },
        lambda prompt: '{"routes":["documents"]}',
    )

    assert result["routes"] == ["documents"]
    assert result["route"] == "documents"


def test_supervisor_uses_llm_for_questions_without_route_keywords():
    result = choose_route(
        {
            "question": "My parcel has not arrived; what should I do?",
            "evidence": {},
            "completed_routes": [],
            "iteration_count": 0,
        },
        lambda prompt: '{"routes":["documents"]}',
    )

    assert result["routes"] == ["documents"]
    assert result["route"] == "documents"


def test_supervisor_builds_structured_plan_for_mixed_question():
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt)
        return (
            '{"routes":["sql","documents"],'
            '"sub_queries":{"sql":"What were last quarter sales?",'
            '"documents":"What are the refund rules?"},'
            '"reason":"Need sales data and policy details.",'
            '"requires_multiple_sources":true,'
            '"confidence":0.96}'
        )

    result = choose_route(
        {
            "question": (
                "Compare last quarter sales with the return policy and refund "
                "rules."
            ),
            "evidence": {},
            "completed_routes": [],
            "pending_routes": [],
            "iteration_count": 0,
        },
        fake_llm,
    )
    assert result["route"] == "sql"
    assert result["routes"] == ["sql", "documents"]
    assert result["active_query"] == "What were last quarter sales?"
    assert result["sub_queries"]["sql"] == "What were last quarter sales?"
    assert result["sub_queries"]["documents"] == "What are the refund rules?"
    assert result["requires_multiple_sources"] is True
    assert result["iteration_count"] == 1
    assert "natural-language task" in prompts[0]
    assert "specialist alone generates SQL" in prompts[0]


def test_sql_shaped_supervisor_subquery_falls_back_to_original_question(
    monkeypatch,
):
    question = "Which store has the highest return rate?"
    invalid_subquery = (
        "SELECT store_id FROM sales_returns "
        "GROUP BY store_id ORDER BY return_rate DESC LIMIT 1"
    )
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt)
        if "You are the supervisor for a retail question-answering system." in prompt:
            return (
                '{"routes":["sql"],"sub_queries":{"sql":'
                + json.dumps(invalid_subquery)
                + "}}"
            )
        if "Generate exactly ONE valid MySQL SELECT query" in prompt:
            return "SELECT st.store_id FROM sales_transactions st LIMIT 1"
        raise AssertionError(f"Unexpected LLM prompt: {prompt}")

    monkeypatch.setattr(
        specialists,
        "execute_sql",
        SimpleNamespace(invoke=lambda _: [{"store_id": "ST-001"}]),
    )
    state = {
        "question": question,
        "evidence": {},
        "completed_routes": [],
        "iteration_count": 0,
    }

    route_update = choose_route(state, fake_llm)
    specialist_update = specialists.run_sql_specialist(
        {**state, **route_update}, fake_llm
    )

    sql_prompt = prompts[1]
    sql_question = sql_prompt.rsplit("QUESTION:\n", maxsplit=1)[1].strip()
    assert route_update["active_query"] == question
    assert route_update["sub_queries"]["sql"] == question
    assert sql_question == question
    assert invalid_subquery not in sql_question
    assert specialist_update["evidence"]["sql"]["status"] == "success"


def test_supervisor_finishes_after_all_specialists():
    result = choose_route(
        {
            "question": "Summarize this",
            "evidence": {"sql": {}, "documents": {}},
            "completed_routes": ["sql", "documents"],
            "pending_routes": [],
            "iteration_count": 2,
        },
        lambda prompt: "should not be called",
    )
    assert result["route"] == "finish"


def test_compiled_graph_runs_both_decomposed_queries(monkeypatch):
    observed = {"sql_queries": [], "document_queries": []}

    def fake_llm(prompt):
        if (
            "You are the supervisor for a retail question-answering system."
            in prompt
        ):
            return (
                '{"routes":["sql","documents"],'
                '"sub_queries":{"sql":"Sales last quarter",'
                '"documents":"Return policy for damaged items"},'
                '"requires_multiple_sources":true}'
            )
        if "You are the supervisor reviewing specialist evidence." in prompt:
            return '{"routes":["documents"]}'
        if "Generate exactly ONE valid MySQL SELECT query" in prompt:
            return "SELECT SUM(amount) FROM sales_transactions"
        if "Answer the user's question using only" in prompt:
            return (
                "Sales totaled 1200; damaged-item returns follow the policy."
            )
        raise AssertionError(f"Unexpected LLM prompt: {prompt}")

    def fake_sql(sql):
        observed["sql_queries"].append(sql)
        return [{"total": 1200}]

    def fake_documents(input_data):
        observed["document_queries"].append(input_data)
        return "[Source: return_policy.pdf]\nDamaged items may be returned."

    monkeypatch.setattr(agent_graph, "call_llm", fake_llm)
    monkeypatch.setattr(
        specialists, "execute_sql", SimpleNamespace(invoke=fake_sql)
    )
    monkeypatch.setattr(
        specialists, "search_documents", SimpleNamespace(invoke=fake_documents)
    )

    result = agent_graph.agent.invoke(
        {
            "question": (
                "Compare sales last quarter with the damaged-item policy."
            ),
            "tenant_id": "tenant-a",
            "user_id": "user-a",
            "evidence": {},
            "completed_routes": [],
            "iteration_count": 0,
        },
        config=conversation_config(f"test-product-category-{uuid4()}"),
    )

    assert observed["sql_queries"] == [
        "SELECT SUM(amount) FROM sales_transactions"
    ]
    assert observed["document_queries"] == [
        {
            "query": "Return policy for damaged items",
            "tenant_id": "tenant-a",
            "user_id": "user-a",
        }
    ]
    assert result["completed_routes"] == ["sql", "documents"]
    assert result["answer"].startswith("Sales totaled 1200")


def test_product_category_revenue_runs_sql_only(monkeypatch):
    seen_prompts = []
    executed_queries = []

    def fake_llm(prompt):
        seen_prompts.append(prompt)
        if (
            "You are the supervisor for a retail question-answering system."
            in prompt
        ):
            return '{"routes":["sql"]}'
        if "Generate exactly ONE valid MySQL SELECT query" in prompt:
            return (
                "SELECT p.category, SUM(st.units_sold * st.unit_price) "
                "AS revenue FROM sales_transactions st "
                "JOIN products p ON p.product_id = st.product_id "
                "GROUP BY p.category ORDER BY revenue DESC LIMIT 5"
            )
        if "Answer the user's question using only" in prompt:
            return "Here are the five highest-revenue product categories."
        raise AssertionError(f"Unexpected LLM prompt: {prompt}")

    def fake_sql(sql):
        executed_queries.append(sql)
        return [{"category": "Groceries", "revenue": 116329.79}]

    def fail_documents(_):
        raise AssertionError("SQL analytics must not call document retrieval")

    monkeypatch.setattr(agent_graph, "call_llm", fake_llm)
    monkeypatch.setattr(
        specialists, "execute_sql", SimpleNamespace(invoke=fake_sql)
    )
    monkeypatch.setattr(
        specialists, "search_documents", SimpleNamespace(invoke=fail_documents)
    )

    result = agent_graph.agent.invoke(
        {
            "question": "Show top 5 product categories by revenue",
            "evidence": {},
            "completed_routes": [],
            "iteration_count": 0,
        },
        config=conversation_config(f"test-product-category-{uuid4()}"),
    )

    assert len(executed_queries) == 1
    assert "LIMIT 5" in executed_queries[0]
    supervisor_prompt_marker = (
        "You are the supervisor for a retail question-answering system."
    )
    assert len(seen_prompts) == 3
    assert supervisor_prompt_marker in seen_prompts[0]
    assert "Generate exactly ONE valid MySQL SELECT query" in seen_prompts[1]
    assert result["completed_routes"] == ["sql"]
    assert result["evidence"].keys() == {"sql"}


def test_non_select_response_is_logged_and_marked_failed(monkeypatch, caplog):
    generated_sql = "The available schema does not support this query."

    def reject_sql(_):
        raise AssertionError("A non-SELECT response must not be executed")

    monkeypatch.setattr(
        specialists, "execute_sql", SimpleNamespace(invoke=reject_sql)
    )
    caplog.set_level("INFO", logger="src.agents.specialists")

    result = specialists.run_sql_specialist(
        {"question": "Which store has the highest return rate?"},
        lambda _: generated_sql,
    )

    assert result["evidence"]["sql"]["status"] == "failed"
    assert any(
        record.message == "sql_query_generated"
        and record.generated_sql == generated_sql
        for record in caplog.records
    )


def test_sql_specialist_returns_structured_order_rate_evidence(monkeypatch):
    generated_sql = (
        "SELECT s.store_name, "
        "COUNT(DISTINCT r.order_id) / COUNT(DISTINCT st.order_id) "
        "* 100 AS order_return_rate_pct "
        "FROM sales_transactions st "
        "JOIN stores s ON s.store_id = st.store_id "
        "LEFT JOIN returns r ON r.order_id = st.order_id "
        "GROUP BY s.store_id, s.store_name "
        "ORDER BY order_return_rate_pct DESC LIMIT 1"
    )
    prompts = []
    executed_queries = []

    def fake_sql(sql):
        executed_queries.append(sql)
        return [{"store_name": "Store A", "order_return_rate_pct": 12.5}]

    monkeypatch.setattr(
        specialists, "execute_sql", SimpleNamespace(invoke=fake_sql)
    )

    result = specialists.run_sql_specialist(
        {"question": "Which store has the highest return rate?"},
        lambda prompt: prompts.append(prompt) or generated_sql,
    )

    sql_evidence = result["evidence"]["sql"]
    assert len(prompts) == 1
    assert "documented relationships" in prompts[0]
    assert "multi-table JOINs" in prompts[0]
    assert "order-based rate" in prompts[0]
    assert "returned-quantity column" in prompts[0]
    assert "sales_transactions.order_id = returns.order_id" in prompts[0]
    assert executed_queries == [generated_sql]
    assert sql_evidence == {
        "status": "success",
        "query": generated_sql,
        "rows": [{"store_name": "Store A", "order_return_rate_pct": 12.5}],
    }


@pytest.mark.parametrize("status", ["unsupported", "failed"])
def test_finalizer_receives_sql_status_and_reason(status):
    prompts = []
    result = finalize(
        {
            "question": "Which store has the highest return rate?",
            "evidence": {
                "sql": {
                    "status": status,
                    "reason": "The requested returned-unit quantity is absent.",
                }
            },
        },
        lambda prompt: prompts.append(prompt)
        or "The calculation is unavailable.",
    )

    assert result["answer"] == "The calculation is unavailable."
    assert f"'status': '{status}'" in prompts[0]
    assert "unsupported or failed" in prompts[0]
    assert "returned-unit quantity is absent" in prompts[0]


def test_finalizer_interprets_successful_sql_rows_directly():
    prompts = []
    result = finalize(
        {
            "question": "Which region had the highest revenue?",
            "evidence": {
                "sql": {
                    "status": "success",
                    "query": "SELECT region, SUM(revenue) FROM sales GROUP BY region",
                    "rows": [{"region": "North", "revenue": 1200}],
                }
            },
        },
        lambda prompt: prompts.append(prompt) or "North had the highest revenue.",
    )

    assert result["answer"] == "North had the highest revenue."
    assert "'status': 'success'" in prompts[0]
    assert "'rows': [{'region': 'North', 'revenue': 1200}]" in prompts[0]
    assert "interpret the query and rows directly" in prompts[0]
    assert "Use specialist analysis as the primary interpretation" not in prompts[0]


def test_timed_llm_call_logs_operation_and_duration(monkeypatch, caplog):
    monkeypatch.setattr(agent_graph, "call_llm", lambda prompt: "result")
    caplog.set_level("INFO", logger="src.agents.graph")

    assert agent_graph._timed_llm_call("prompt", "sql_generation") == "result"
    timing = next(
        record for record in caplog.records
        if record.message == "llm_call_timing"
    )
    assert timing.operation == "sql_generation"
    assert timing.status == "success"
    assert timing.duration_ms >= 0


def test_destructive_request_is_refused_without_sql_execution(monkeypatch):
    def fake_llm(prompt):
        if (
            "You are the supervisor for a retail question-answering system."
            in prompt
        ):
            return '{"routes":["sql"]}'
        raise AssertionError("Destructive requests must not call the SQL LLM")

    def fail_sql(_):
        raise AssertionError("Destructive requests must not execute SQL")

    monkeypatch.setattr(agent_graph, "call_llm", fake_llm)
    monkeypatch.setattr(
        specialists, "execute_sql", SimpleNamespace(invoke=fail_sql)
    )

    result = agent_graph.agent.invoke(
        {
            "question": "Delete all sales records.",
            "evidence": {},
            "completed_routes": [],
            "iteration_count": 0,
        },
        config=conversation_config(f"test-destructive-request-{uuid4()}"),
    )

    assert result["evidence"]["sql"]["status"] == "rejected"
    assert "can’t modify or delete" in result["answer"]


def test_checkpointed_follow_up_uses_prior_answer_and_evidence(
    monkeypatch, tmp_path
):
    observed = {"sql": 0, "document": 0, "supervisor_prompts": []}
    connection = sqlite3.connect(
        tmp_path / "conversation.sqlite", check_same_thread=False
    )
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()
    agent = agent_graph.graph.compile(checkpointer=checkpointer)

    def fake_llm(prompt):
        if (
            "You are the supervisor for a retail question-answering system."
            in prompt
        ):
            observed["supervisor_prompts"].append(prompt)
            if "What data did you use for that answer?" in prompt:
                return '{"answer_from_history":true,"routes":[]}'
            return '{"routes":["sql"]}'
        if "Generate exactly ONE valid MySQL SELECT query" in prompt:
            return "SELECT SUM(revenue) FROM sales_transactions"
        if "Answer the user's question using only" in prompt:
            if "What data did you use for that answer?" in prompt:
                return "I used the SQL result: revenue totaled 1200."
            return "Category revenue totaled 1200."
        raise AssertionError(f"Unexpected LLM prompt: {prompt}")

    def fake_sql(_sql):
        observed["sql"] += 1
        return [{"revenue": 1200}]

    def fail_documents(_query):
        observed["document"] += 1
        raise AssertionError(
            "History-only follow-up must not search documents"
        )

    monkeypatch.setattr(agent_graph, "call_llm", fake_llm)
    monkeypatch.setattr(
        specialists, "execute_sql", SimpleNamespace(invoke=fake_sql)
    )
    monkeypatch.setattr(
        specialists, "search_documents", SimpleNamespace(invoke=fail_documents)
    )
    config = conversation_config("memory-test")
    base_state = {
        "evidence": {},
        "route": "",
        "answer": "",
        "routes": [],
        "sub_queries": {},
        "active_query": "",
        "plan_initialized": False,
        "completed_routes": [],
        "pending_routes": [],
        "iteration_count": 0,
    }

    try:
        first = agent.invoke(
            {**base_state, "question": "Summarize revenue by category."},
            config=config,
        )
        second = agent.invoke(
            {
                **base_state,
                "question": "What data did you use for that answer?",
            },
            config=config,
        )
    finally:
        connection.close()

    assert first["answer"] == "Category revenue totaled 1200."
    assert "SQL result" in second["answer"]
    assert observed["sql"] == 1
    assert observed["document"] == 0
    assert (
        "Category revenue totaled 1200."
        in observed["supervisor_prompts"][1]
    )
    assert "revenue" in observed["supervisor_prompts"][1]
    assert "1200" in observed["supervisor_prompts"][1]


def test_remember_turn_keeps_compact_sql_and_document_evidence():
    from src.agents.supervisor import _remember_turn

    history = _remember_turn(
        {
            "question": "Summarize South-region sales.",
            "evidence": {
                "sql": {
                    "query": "SELECT total FROM sales WHERE region = 'South'",
                    "status": "success",
                    "rows": [{"total": 1200}, {"total": 800}],
                },
                "documents": (
                    "[Source: return_policy.txt]\n"
                    + "Policy excerpt. " * 100
                    + "\n[Source: faq.txt]\nSecond excerpt."
                ),
            },
        },
        "South-region sales totaled 1200.",
    )

    evidence = history[0]["evidence"]
    assert "region = 'South'" in evidence["sql"]["query"]
    assert evidence["sql"]["status"] == "success"
    assert evidence["sql"]["result_sample"]
    assert "summary" not in evidence["sql"]
    assert len(evidence["documents"]) == 2
    assert evidence["documents"][0]["source"] == "return_policy.txt"
    assert len(evidence["documents"][0]["excerpt"]) < 300


def test_follow_up_keeps_prior_region_filter_in_online_sql_query(
    monkeypatch, tmp_path
):
    observed = {
        "sql_prompts": [],
        "executed_sql": [],
        "supervisor_prompts": [],
    }
    connection = sqlite3.connect(
        tmp_path / "filter-follow-up.sqlite", check_same_thread=False
    )
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()
    agent = agent_graph.graph.compile(checkpointer=checkpointer)

    def fake_llm(prompt):
        if (
            "You are the supervisor for a retail question-answering system."
            in prompt
        ):
            observed["supervisor_prompts"].append(prompt)
            if "only by online channel" in prompt:
                assert "What were sales in South region?" in prompt
                return (
                    '{"routes":["sql"],"sub_queries":{"sql":'
                    '"sales generated in South region by online channel"}}'
                )
            return (
                '{"routes":["sql"],"sub_queries":{"sql":'
                '"sales generated in South region"}}'
            )
        if "Generate exactly ONE valid MySQL SELECT query" in prompt:
            observed["sql_prompts"].append(prompt)
            if "online channel" in prompt:
                return (
                    "SELECT SUM(amount) FROM sales_transactions "
                    "WHERE region = 'South' AND channel = 'Online'"
                )
            return (
                "SELECT SUM(amount) FROM sales_transactions "
                "WHERE region = 'South'"
            )
        if "Answer the user's question using only" in prompt:
            return "Online sales in the South totaled 700."
        raise AssertionError(f"Unexpected LLM prompt: {prompt}")

    def fake_sql(sql):
        observed["executed_sql"].append(sql)
        return [{"total": 700 if "Online" in sql else 1200}]

    def fail_documents(_query):
        raise AssertionError("This SQL follow-up must not search documents")

    monkeypatch.setattr(agent_graph, "call_llm", fake_llm)
    monkeypatch.setattr(
        specialists, "execute_sql", SimpleNamespace(invoke=fake_sql)
    )
    monkeypatch.setattr(
        specialists,
        "search_documents",
        SimpleNamespace(invoke=fail_documents),
    )
    config = conversation_config("south-online-follow-up")
    base_state = {
        "evidence": {},
        "route": "",
        "answer": "",
        "routes": [],
        "sub_queries": {},
        "active_query": "",
        "plan_initialized": False,
        "completed_routes": [],
        "pending_routes": [],
        "iteration_count": 0,
    }

    try:
        agent.invoke(
            {**base_state, "question": "What were sales in South region?"},
            config=config,
        )
        agent.invoke(
            {
                **base_state,
                "question": (
                    "Now, what were sales generated only by online channel?"
                ),
            },
            config=config,
        )
    finally:
        connection.close()

    assert len(observed["sql_prompts"]) == 2
    follow_up_prompt = observed["sql_prompts"][1]
    assert "South region" in follow_up_prompt
    assert "online channel" in follow_up_prompt
    assert "region = 'South'" in observed["executed_sql"][1]
    assert "channel = 'Online'" in observed["executed_sql"][1]
    assert "South region" in observed["supervisor_prompts"][1]
