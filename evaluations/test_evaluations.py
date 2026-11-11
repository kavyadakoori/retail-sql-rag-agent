from __future__ import annotations

import importlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest
from openpyxl import Workbook, load_workbook

from src.agents.graph import clear_conversation, conversation_config
from src.config import Config
from src.tools.sql_safety import is_destructive_request, validate_sql

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_CASES_PATH = PROJECT_ROOT / "evaluations" / "test_cases.xlsx"
EVAL_REPORT_PATH = (
    PROJECT_ROOT / "evaluations" / "test_cases_evaluation_results.xlsx"
)
EVAL_SHEET_NAME = "test_cases"
SOURCE_PATTERN = re.compile(r"\[Source:\s*([^\]]+)\]")
RAGAS_VERTEX_MODULE = "langchain_community.chat_models.vertexai"
REQUIRED_COLUMNS = {
    "Test ID",
    "Question",
    "Expected Answer/Ground truth",
    "Expected routes",
    "Expected docs",
}


def _route_value(case: dict[str, Any]) -> Any:
    for key in ("chosen routes", "Expected routes", "Expected Route"):
        if key in case and case.get(key) is not None:
            return case.get(key)
    return None


def _docs_value(case: dict[str, Any]) -> Any:
    for key in ("Chosen docs", "Expected docs", "Relevant Docs"):
        if key in case and case.get(key) is not None:
            return case.get(key)
    return None


def _expected_routes(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        values = [str(item) for item in value]
    else:
        text = str(value).strip().casefold()
        if text in {"", "none", "no route", "n/a", "na", "-", "—"}:
            return []
        values = [text]

    routes = []
    for item in values:
        for route in re.findall(r"\b(sql|docs?|documents)\b", item.casefold()):
            normalized = (
                "documents"
                if route in {"doc", "docs", "document", "documents"}
                else "sql"
            )
            if normalized not in routes:
                routes.append(normalized)
    return routes


def _expected_sources(value: Any) -> set[str]:
    if value is None:
        return set()
    text = str(value).strip()
    if text.casefold() in {"", "none", "n/a", "na"}:
        return set()
    return {
        source.strip().casefold()
        for source in re.split(r"[,;]", text)
        if source.strip()
    }


def _load_eval_cases() -> list[dict[str, Any]]:
    workbook = load_workbook(EVAL_CASES_PATH, data_only=True, read_only=True)
    try:
        worksheet = workbook[EVAL_SHEET_NAME]
        values = worksheet.iter_rows(values_only=True)
        headers = [
            str(value).strip() if value is not None else ""
            for value in next(values)
        ]
        cases = [
            {
                header: value if value is not None else ""
                for header, value in zip(headers, row)
                if header
            }
            for row in values
            if any(value is not None and str(value).strip() for value in row)
        ]
    finally:
        workbook.close()

    for case in cases:
        raw_routes = _route_value(case)
        raw_sources = _docs_value(case)
        case["expected_routes"] = _expected_routes(raw_routes)
        case["expected_sources"] = _expected_sources(raw_sources)
        case["chosen routes"] = raw_routes
        case["Chosen docs"] = raw_sources
    return cases


def _validate_workbook_cases(cases: list[dict[str, Any]]) -> list[str]:
    errors = []
    if not cases:
        return ["Workbook has no evaluation cases"]
    missing_columns = REQUIRED_COLUMNS - set(cases[0])
    if missing_columns:
        errors.append(
            "Missing required columns: " + ", ".join(sorted(missing_columns))
        )
        return errors

    seen_ids = set()
    for case in cases:
        test_id = str(case.get("Test ID") or "").strip()
        if not test_id:
            errors.append("A workbook row is missing Test ID")
            continue
        if test_id in seen_ids:
            errors.append(f"{test_id}: duplicate Test ID")
        seen_ids.add(test_id)
        if not str(case.get("Question") or "").strip():
            errors.append(f"{test_id}: Question is empty")
        if not str(case.get("Expected Answer/Ground truth") or "").strip():
            errors.append(f"{test_id}: Expected Answer/Ground truth is empty")
        raw_routes = _route_value(case)
        route_text = str(raw_routes or "").strip().casefold()
        no_route_labels = {"", "none", "no route", "n/a", "na", "-", "—"}
        has_expected_route = bool(case.get("expected_routes"))
        if route_text not in no_route_labels and not has_expected_route:
            errors.append(f"{test_id}: chosen routes is empty or unrecognized")
        if "documents" in case.get("expected_routes", []) and not case.get(
            "expected_sources"
        ):
            errors.append(
                f"{test_id}: Chosen docs is required for document cases"
            )
    return errors


def _route_scores(
    expected_routes: list[str], actual_routes: list[str]
) -> dict[str, float | bool]:
    expected = set(expected_routes)
    actual = set(actual_routes)
    overlap = expected & actual
    precision = len(overlap) / len(actual) if actual else 0.0
    recall = len(overlap) / len(expected) if expected else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "match": expected == actual,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _retrieved_sources(context: str) -> set[str]:
    return {
        source.strip().casefold()
        for source in SOURCE_PATTERN.findall(context or "")
    }


def _document_context_chunks(context: str) -> list[str]:
    matches = list(SOURCE_PATTERN.finditer(context or ""))
    if not matches:
        return [context.strip()] if context and context.strip() else []
    chunks = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else None
        chunk = context[match.end():end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _sql_evaluation_context(sql_evidence: dict[str, Any]) -> str:
    if not sql_evidence:
        return ""
    rows = sql_evidence.get("rows", [])
    parts = (
        f"Generated SQL: {sql_evidence.get('query', '')}",
        f"SQL analysis: {sql_evidence.get('analysis', '')}",
        f"Returned row sample: {json.dumps(rows[:3], default=str)}",
    )
    return "\n".join(part for part in parts if part.strip())


def _history_evaluation_context(history: list[dict[str, Any]]) -> list[str]:
    contexts = []
    for turn in history:
        question = str(turn.get("question", ""))
        answer = str(turn.get("answer", ""))
        if question or answer:
            contexts.append(
                f"Prior question: {question}\nPrior answer: {answer}"
            )
        evidence = turn.get("evidence", {})
        if isinstance(evidence, str):
            try:
                evidence = json.loads(evidence)
            except json.JSONDecodeError:
                evidence = {}
        if not isinstance(evidence, dict):
            continue
        sql = evidence.get("sql", {})
        if isinstance(sql, dict):
            sql_text = _sql_evaluation_context(sql)
            if sql_text:
                contexts.append(sql_text)
        documents = evidence.get("documents", [])
        if isinstance(documents, list):
            for document in documents:
                if isinstance(document, dict) and document.get("excerpt"):
                    contexts.append(
                        f"Document excerpt: {document['excerpt']}"
                    )
        elif documents:
            contexts.extend(_document_context_chunks(str(documents)))
    return contexts


def _ensure_ragas_import_compatibility() -> None:
    try:
        importlib.import_module(RAGAS_VERTEX_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != RAGAS_VERTEX_MODULE:
            raise
        compatibility_module = type(sys)(RAGAS_VERTEX_MODULE)
        compatibility_module.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[RAGAS_VERTEX_MODULE] = compatibility_module


def _extract_ragas_metric_scores(
    values: dict[str, Any], metric_names: tuple[str, ...]
) -> dict[str, float]:
    scores = {}
    for name in metric_names:
        value = values.get(name)
        if value is None:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            scores[name] = score
    return scores


def _run_ragas_metrics(
    question: str,
    answer: str,
    evidence_contexts: list[str],
    ground_truth: str,
    document_contexts: list[str],
) -> dict[str, float]:
    _ensure_ragas_import_compatibility()
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=Config.AI_GATEWAY_MODEL,
            api_key=Config.AI_GATEWAY_API_KEY,
            base_url=Config.AI_GATEWAY_URL,
            temperature=0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=Config.AI_GATEWAY_API_KEY,
            base_url=Config.AI_GATEWAY_URL,
        )
    )

    def score(
        contexts: list[str],
        metrics: list[Any],
        metric_names: tuple[str, ...],
    ) -> dict[str, float]:
        dataset = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth],
            }
        )
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            show_progress=False,
            raise_exceptions=False,
        )
        values = result.to_pandas().iloc[0].to_dict()
        return _extract_ragas_metric_scores(values, metric_names)

    scores = score(
        evidence_contexts,
        [
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm, embeddings=embeddings),
        ],
        ("faithfulness", "answer_relevancy"),
    )
    if document_contexts:
        scores.update(
            score(
                document_contexts,
                [ContextPrecision(llm=llm), ContextRecall(llm=llm)],
                ("context_precision", "context_recall"),
            )
        )
    return scores


def _numeric_mean(results: list[dict[str, Any]], field: str):
    values = []
    for result in results:
        value = result.get(field)
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            values.append(float(value))
        elif isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return (sum(values) / len(values), len(values)) if values else (0.0, 0)


def _summarize_results(
    results: list[dict[str, Any]],
) -> list[tuple[str, float | int, int]]:
    metrics = [("Cases evaluated", len(results), len(results))]
    for field, label in (
        ("Route Match", "Exact route-set accuracy"),
        ("SQL Executable", "SQL executable-query rate"),
        ("SQL Safety Passed", "SQL safety pass rate"),
        ("Ragas Context Precision", "Ragas context precision"),
        ("Ragas Context Recall", "Ragas context recall"),
        ("Ragas Faithfulness", "Ragas answer faithfulness"),
        ("Ragas Answer Relevancy", "Ragas answer relevancy"),
    ):
        score, count = _numeric_mean(results, field)
        metrics.append((label, score, count))
    return metrics


def _targeted_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_ids = os.getenv("EVAL_TARGET_IDS", "").strip()
    if not target_ids:
        return cases
    allowed_ids = {
        test_id.strip()
        for test_id in target_ids.split(",")
        if test_id.strip()
    }
    return [
        case
        for case in cases
        if str(case.get("Test ID", "")).strip() in allowed_ids
    ]


def _write_report(results: list[dict[str, Any]]) -> None:
    if not EVAL_REPORT_PATH.exists():
        shutil.copy2(EVAL_CASES_PATH, EVAL_REPORT_PATH)
    workbook = load_workbook(EVAL_REPORT_PATH)
    try:
        worksheet = workbook[EVAL_SHEET_NAME]
        headers = {
            cell.value: cell.column for cell in worksheet[1] if cell.value
        }
        output_fields = [
            "Expected Routes",
            "Actual Routes",
            "Route Match",
            "Route Precision",
            "Route Recall",
            "Route F1",
            "SQL Executed",
            "SQL Safety Passed",
            "SQL Executable",
            "Retrieved Docs",
            "Retrieved Chunks",
            "Evaluation Contexts",
            "Ragas Context Precision",
            "Ragas Context Recall",
            "Ragas Faithfulness",
            "Ragas Answer Relevancy",
            "Evaluation Error",
        ]
        generated_fields = [
            "Generated Answer",
            "Generated sql query",
        ]
        for field in output_fields + generated_fields:
            if field not in headers:
                column = worksheet.max_column + 1
                worksheet.cell(1, column, field)
                headers[field] = column

        rows_by_id = {
            str(worksheet.cell(row, headers["Test ID"]).value): row
            for row in range(2, worksheet.max_row + 1)
        }
        generated_answer_column = headers["Generated Answer"]
        generated_sql_column = headers["Generated sql query"]
        for result in results:
            row_number = rows_by_id[str(result["Test ID"])]
            for field in output_fields:
                value = result.get(field, "")
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=True, default=str)
                worksheet.cell(row_number, headers[field], value)
            worksheet.cell(
                row_number,
                generated_answer_column,
                result.get("Generated Answer", ""),
            )
            worksheet.cell(
                row_number,
                generated_sql_column,
                result.get("Generated sql query", ""),
            )

        summary = (
            workbook["Summary"]
            if "Summary" in workbook.sheetnames
            else workbook.create_sheet("Summary")
        )
        summary.delete_rows(1, summary.max_row)
        summary.append(["Evaluation Metric", "Score", "Cases"])
        score_fields = (
            "Route Match",
            "SQL Executable",
            "SQL Safety Passed",
            "Ragas Context Precision",
            "Ragas Context Recall",
            "Ragas Faithfulness",
            "Ragas Answer Relevancy",
        )
        scored_results = []
        for row_number in range(2, worksheet.max_row + 1):
            scored_result = {
                field: worksheet.cell(row_number, headers[field]).value
                for field in score_fields
            }
            if any(
                value is not None and value != ""
                for value in scored_result.values()
            ):
                scored_results.append(scored_result)
        for metric, score, count in _summarize_results(scored_results):
            summary.append([metric, score, count])
        workbook.save(EVAL_REPORT_PATH)
    finally:
        workbook.close()


def _evaluate_case(
    case: dict[str, Any], conversation_id: str | None = None
) -> dict[str, Any]:
    from src.agents import specialists
    from src.agents.graph import agent

    conversation_id = conversation_id or f"case-{uuid4()}"
    config = conversation_config(conversation_id)
    previous_history = []
    get_state = getattr(agent, "get_state", None)
    if callable(get_state):
        try:
            previous_history = get_state(config).values.get(
                "conversation_history", []
            )
        except (AttributeError, KeyError, TypeError):
            previous_history = []

    state = {
        "question": str(case["Question"]),
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
    attempted_sql = []

    def execute_and_capture(sql: str):
        attempted_sql.append(sql)
        return original_executor.invoke(sql)

    original_executor = specialists.execute_sql
    routes = []
    evidence = {}
    answer = ""
    answer_from_history = False
    error = ""
    try:
        with patch.object(
            specialists,
            "execute_sql",
            type(
                "Executor",
                (),
                {"invoke": staticmethod(execute_and_capture)},
            )(),
        ):
            for update in agent.stream(
                state, config=config, stream_mode="updates"
            ):
                for node, node_update in update.items():
                    if node == "supervisor":
                        route = node_update.get("route")
                        if route in {"sql", "documents"}:
                            routes.append(route)
                        answer_from_history = answer_from_history or bool(
                            node_update.get("answer_from_history")
                        )
                    if node_update.get("evidence"):
                        evidence.update(node_update["evidence"])
                    if node_update.get("answer"):
                        answer = node_update["answer"]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    sql_evidence = evidence.get("sql", {})
    document_context = str(evidence.get("documents", ""))
    generated_sql = sql_evidence.get("query") or (
        attempted_sql[-1] if attempted_sql else ""
    )
    sql_executed = isinstance(sql_evidence, dict) and "rows" in sql_evidence
    expected_routes = case.get("expected_routes") or _expected_routes(
        case.get("chosen routes")
    )
    actual_routes = list(dict.fromkeys(routes))
    route_score = _route_scores(expected_routes, actual_routes)
    destructive = is_destructive_request(str(case["Question"]))

    sql_safety_passed: bool | str = ""
    sql_executable: bool | str = ""
    if "sql" in expected_routes:
        if destructive:
            if generated_sql:
                try:
                    validate_sql(generated_sql)
                    sql_safety_passed = True
                except ValueError:
                    sql_safety_passed = False
            else:
                sql_safety_passed = True
            sql_executable = False
        elif not generated_sql:
            sql_safety_passed = "NOT_GENERATED"
            sql_executable = "NOT_GENERATED"
        else:
            try:
                validate_sql(generated_sql)
                sql_safety_passed = True
            except ValueError:
                sql_safety_passed = False
            sql_executable = bool(sql_safety_passed and sql_executed)

    document_chunks = _document_context_chunks(document_context)
    retrieved_sources = sorted(_retrieved_sources(document_context))
    current_contexts = list(document_chunks)
    sql_context = _sql_evaluation_context(sql_evidence)
    if sql_context:
        current_contexts.append(sql_context)
    history_contexts = (
        _history_evaluation_context(previous_history)
        if answer_from_history
        else []
    )
    evaluation_contexts = history_contexts + current_contexts
    expected_documents = "documents" in expected_routes

    checks = []
    if not route_score["match"] and not (
        destructive and sql_safety_passed is True
    ):
        checks.append(
            f"actual routes {actual_routes!r} did not match expected "
            f"routes {expected_routes!r}"
        )
    if "sql" in expected_routes:
        if destructive and not sql_safety_passed:
            checks.append("destructive SQL request was not rejected")
        elif not destructive and not sql_executable:
            checks.append("SQL was not both safe and successfully executed")
    if "documents" in expected_routes and not document_context:
        checks.append("document retrieval returned no context")
    if error and not (destructive and sql_safety_passed):
        checks.append(error)
    if not answer and not (destructive and sql_safety_passed):
        checks.append("no final answer was produced")

    ragas_scores = {}
    ground_truth = str(case.get("Expected Answer/Ground truth") or "").strip()
    
    # Live LLM judges are disabled by default to save API quotas
    run_live_judges = os.getenv("RUN_LIVE_JUDGES", "0") == "1"
    if run_live_judges and answer and ground_truth and evaluation_contexts:
        try:
            ragas_scores = _run_ragas_metrics(
                str(case["Question"]),
                answer,
                evaluation_contexts,
                ground_truth,
                document_chunks if expected_documents else [],
            )
        except Exception as exc:
            checks.append(
                f"Ragas evaluation failed: {type(exc).__name__}: {exc}"
            )
    if run_live_judges and expected_documents and document_chunks and ground_truth:
        for metric in ("context_precision", "context_recall"):
            if metric not in ragas_scores:
                checks.append(f"Ragas did not return {metric}")

    return {
        "Test ID": case["Test ID"],
        "Expected Routes": expected_routes,
        "Actual Routes": actual_routes,
        "Route Match": route_score["match"],
        "Route Precision": route_score["precision"],
        "Route Recall": route_score["recall"],
        "Route F1": route_score["f1"],
        "SQL Executed": sql_executed,
        "SQL Safety Passed": sql_safety_passed,
        "SQL Executable": sql_executable,
        "Retrieved Docs": retrieved_sources,
        "Retrieved Chunks": document_chunks,
        "Evaluation Contexts": evaluation_contexts,
        "Generated Answer": answer,
        "Generated sql query": generated_sql,
        "Ragas Context Precision": ragas_scores.get("context_precision", ""),
        "Ragas Context Recall": ragas_scores.get("context_recall", ""),
        "Ragas Faithfulness": ragas_scores.get("faithfulness", ""),
        "Ragas Answer Relevancy": ragas_scores.get("answer_relevancy", ""),
        "Evaluation Error": "; ".join(checks),
        "checks": checks,
    }


@pytest.mark.skipif(
    os.getenv("RUN_AGENT_EVALUATIONS") != "1",
    reason=(
        "Set RUN_AGENT_EVALUATIONS=1 to run workbook cases against MySQL, "
        "Qdrant, and the configured LLM gateway."
    ),
)
def test_agent_against_workbook_ground_truth():
    cases = _load_eval_cases()
    workbook_errors = _validate_workbook_cases(cases)
    assert not workbook_errors, "Invalid workbook:\n" + "\n".join(
        workbook_errors
    )

    results = []
    failures = []
    # Dynamic session grouping: keep conversational continuity only for follow-ups
    follow_up_ids = {"TC-006", "TC-010"} 
    current_conversation_id = f"evaluation-{uuid4()}"
    targeted_cases = _targeted_cases(cases)
    try:
        for case in targeted_cases:
            test_id = str(case.get("Test ID", "")).strip()
            if test_id not in follow_up_ids:
                current_conversation_id = f"evaluation-{uuid4()}"

            result = _evaluate_case(case, current_conversation_id)
            results.append(result)

            if result["checks"]:
                failures.append(
                    f"{case['Test ID']}: " + "; ".join(result["checks"])
                )
    finally:
        try:
            _write_report(results)
        finally:
            clear_conversation(current_conversation_id)

    assert not failures, "Evaluation failures:\n" + "\n".join(failures)


def test_workbook_ground_truth_preflight():
    cases = _load_eval_cases()

    assert len(cases) == 30
    assert _validate_workbook_cases(cases) == []
    assert _expected_routes(cases[5]["chosen routes"]) == ["sql"]
    assert _expected_routes(cases[9]["chosen routes"]) == []


def test_expected_routes_normalize_docs_and_no_route_labels():
    assert _expected_routes("sql,docs") == ["sql", "documents"]
    assert _expected_routes("None") == []


def test_ragas_score_extraction_ignores_text_columns():
    from evaluations.test_evaluations import _extract_ragas_metric_scores

    result = _extract_ragas_metric_scores(
        {
            "question": "What is our store return policy for electronics?",
            "contexts": ["Relevant policy excerpt"],
            "faithfulness": 0.91,
            "answer_relevancy": 0.87,
            "context_precision": 0.95,
            "context_recall": float("nan"),
        },
        (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ),
    )

    assert result == {
        "faithfulness": 0.91,
        "answer_relevancy": 0.87,
        "context_precision": 0.95,
    }


def test_route_scores_ignore_order_but_penalize_missing_routes():
    assert _route_scores(["documents", "sql"], ["sql", "documents"])[
        "match"
    ]
    scores = _route_scores(["sql", "documents"], ["sql"])
    assert scores["match"] is False
    assert scores["precision"] == 1.0
    assert scores["recall"] == 0.5


def test_document_context_chunks_exclude_source_labels():
    text = (
        "[Source: faq.txt]\nFirst answer.\n\n"
        "[Source: policy.pdf, page 2]\nSecond answer."
    )

    assert _retrieved_sources(text) == {"faq.txt", "policy.pdf, page 2"}
    assert _document_context_chunks(text) == [
        "First answer.",
        "Second answer.",
    ]


def test_history_evaluation_context_labels_document_excerpts():
    assert _history_evaluation_context(
        [
            {
                "evidence": {
                    "documents": [
                        {"excerpt": "A document-backed answer."},
                    ]
                }
            }
        ]
    ) == ["Document excerpt: A document-backed answer."]


def test_evaluation_summary_uses_only_scored_cases():
    summary = dict(
        (name, (score, count))
        for name, score, count in _summarize_results(
            [
                {"Route Match": True, "SQL Executable": True},
                {"Route Match": False, "SQL Executable": ""},
            ]
        )
    )

    assert summary["Exact route-set accuracy"] == (0.5, 2)
    assert summary["SQL executable-query rate"] == (1.0, 1)


def test_targeted_cases_filters_comma_separated_ids(monkeypatch):
    cases = [
        {"Test ID": "TC-001"},
        {"Test ID": "TC-004"},
        {"Test ID": "TC-007"},
    ]
    monkeypatch.setenv("EVAL_TARGET_IDS", " TC-004, TC-007 ")

    assert _targeted_cases(cases) == cases[1:]


def test_write_report_updates_only_requested_rows_and_summarizes_all(
    monkeypatch, tmp_path
):
    report_path = tmp_path / "report.xlsx"
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = EVAL_SHEET_NAME
    worksheet.append(
        [
            "Test ID",
            "Question",
            "Generated Answer",
            "Generated sql query",
            "Route Match",
            "SQL Executable",
            "SQL Safety Passed",
            "Ragas Context Precision",
            "Ragas Context Recall",
            "Ragas Faithfulness",
            "Ragas Answer Relevancy",
        ]
    )
    worksheet.append(
        ["TC-001", "First", "Old answer", "SELECT 1", True, True, True]
    )
    worksheet.append(
        ["TC-002", "Second", "Keep this answer", "SELECT 2", False, False, True]
    )
    workbook.create_sheet("Summary")
    workbook.save(report_path)
    workbook.close()
    monkeypatch.setattr(sys.modules[__name__], "EVAL_REPORT_PATH", report_path)

    _write_report(
        [
            {
                "Test ID": "TC-001",
                "Expected Routes": ["sql"],
                "Actual Routes": ["sql"],
                "Route Match": True,
                "Route Precision": 1.0,
                "Route Recall": 1.0,
                "Route F1": 1.0,
                "SQL Executed": True,
                "SQL Safety Passed": True,
                "SQL Executable": True,
                "Generated Answer": "Updated answer",
                "Generated sql query": "SELECT 3",
                "Evaluation Error": "",
            }
        ]
    )

    saved = load_workbook(report_path, data_only=True)
    try:
        worksheet = saved[EVAL_SHEET_NAME]
        headers = {
            cell.value: cell.column for cell in worksheet[1] if cell.value
        }
        assert worksheet.cell(2, headers["Generated Answer"]).value == (
            "Updated answer"
        )
        assert worksheet.cell(2, headers["Generated sql query"]).value == (
            "SELECT 3"
        )
        assert worksheet.cell(3, headers["Generated Answer"]).value == (
            "Keep this answer"
        )
        assert worksheet.cell(3, headers["Generated sql query"]).value == (
            "SELECT 2"
        )
        summary = saved["Summary"]
        metrics = {
            summary.cell(row, 1).value: (
                summary.cell(row, 2).value,
                summary.cell(row, 3).value,
            )
            for row in range(2, summary.max_row + 1)
        }
        assert metrics["Cases evaluated"] == (2, 2)
        assert metrics["Exact route-set accuracy"] == (0.5, 2)
        assert metrics["SQL executable-query rate"] == (0.5, 2)
    finally:
        saved.close()


def test_destructive_document_route_is_not_reported_as_route_mismatch(
    monkeypatch,
):
    from src.agents import graph as agent_graph

    class RefusedAgent:
        def stream(self, state, config, stream_mode):
            yield {"supervisor": {"route": "documents"}}
            yield {
                "sql_specialist": {
                    "evidence": {
                        "sql": {
                            "status": "rejected",
                            "reason": "Only read-only SQL queries are allowed.",
                        }
                    }
                }
            }
            yield {"finalize": {"answer": "I cannot modify database records."}}

    monkeypatch.setattr(agent_graph, "agent", RefusedAgent())

    result = _evaluate_case(
        {
            "Test ID": "TC-007",
            "Question": "Delete all sales records",
            "expected_routes": ["sql"],
            "Expected Answer/Ground truth": "The request was refused.",
        }
    )

    assert result["Actual Routes"] == ["documents"]
    assert result["SQL Safety Passed"] is True
    assert result["SQL Executable"] is False
    assert not any("did not match expected routes" in check for check in result["checks"])