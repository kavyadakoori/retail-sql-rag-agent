from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from evaluations.test_evaluations import (
    EVAL_REPORT_PATH,
    _expected_routes,
    _run_ragas_metrics,
)

RAGAS_FIELDS = {
    "Ragas Context Precision": "context_precision",
    "Ragas Context Recall": "context_recall",
    "Ragas Faithfulness": "faithfulness",
    "Ragas Answer Relevancy": "answer_relevancy",
}
RAGAS_ERROR_PREFIXES = (
    "Ragas evaluation failed:",
    "Ragas did not return ",
)


def _read_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return []
    if isinstance(decoded, list):
        return [str(item) for item in decoded if item is not None]
    return []


def _as_score(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _clean_ragas_errors(value: Any) -> list[str]:
    if not value:
        return []
    return [
        message.strip()
        for message in str(value).split(";")
        if message.strip()
        and not message.strip().startswith(RAGAS_ERROR_PREFIXES)
    ]


def _decode_contexts(value: Any) -> list[str]:
    return _read_list(value)


def _header_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


HEADER_ALIASES = {
    "Test ID": ("Test ID", "Test Id"),
    "Question": ("Question", "User Question"),
    "Expected Answer/Ground truth": (
        "Expected Answer/Ground truth",
        "Expected Answer/Ground Truth",
        "Expected Answer / Ground truth",
        "Ground truth",
    ),
    "chosen routes": (
        "chosen routes",
        "Expected Routes",
        "Expected routes",
        "Expected Route",
    ),
    "Generated Answer": ("Generated Answer", "Generated answer"),
    "Generated sql query": (
        "Generated sql query",
        "Generated SQL query",
        "Generated SQL Query",
        "Generated SQL",
    ),
    "Retrieved Chunks": ("Retrieved Chunks", "Retrieved chunks"),
    "Evaluation Contexts": ("Evaluation Contexts", "Evaluation contexts"),
    "Evaluation Error": ("Evaluation Error", "Evaluation error"),
}


def _resolve_headers(worksheet) -> dict[str, int]:
    headers = {
        str(cell.value).strip(): cell.column
        for cell in worksheet[1]
        if cell.value
    }
    normalized_headers = {
        _header_key(name): column for name, column in headers.items()
    }
    columns = {}
    fields = (
        "Test ID",
        "Question",
        "Expected Answer/Ground truth",
        "chosen routes",
        "Generated Answer",
        "Generated sql query",
        "Retrieved Chunks",
        "Evaluation Contexts",
        "Evaluation Error",
        *RAGAS_FIELDS,
    )
    for field in fields:
        aliases = HEADER_ALIASES.get(field, (field,))
        column = next(
            (headers[alias] for alias in aliases if alias in headers),
            None,
        )
        if column is None:
            column = next(
                (
                    normalized_headers[_header_key(alias)]
                    for alias in aliases
                    if _header_key(alias) in normalized_headers
                ),
                None,
            )
        if column is not None:
            columns[field] = column
    return columns


def refresh_report(
    report_path: Path = EVAL_REPORT_PATH,
    test_id: str | None = None,
    skip_scored: bool = False,
) -> tuple[int, int]:
    workbook = load_workbook(report_path)
    save_path = report_path
    scored = 0
    skipped = 0
    try:
        worksheet = workbook["test_cases"]
        headers = _resolve_headers(worksheet)
        required = {
            "Test ID",
            "Question",
            "Expected Answer/Ground truth",
            "chosen routes",
            "Generated Answer",
            "Retrieved Chunks",
            "Evaluation Contexts",
            "Evaluation Error",
            *RAGAS_FIELDS,
        }
        missing = required - set(headers)
        if missing:
            raise ValueError(
                "Evaluation report is missing columns: "
                + ", ".join(sorted(missing))
            )

        def read(row_number: int, column: str):
            column_number = headers.get(column)
            return (
                worksheet.cell(row_number, column_number).value
                if column_number is not None
                else None
            )

        def save_workbook() -> None:
            nonlocal save_path
            try:
                workbook.save(save_path)
            except PermissionError:
                if save_path != report_path:
                    raise
                save_path = report_path.with_name(
                    f"{report_path.stem}_ragas_refreshed{report_path.suffix}"
                )
                workbook.save(save_path)
                print(
                    f"Report is locked; saving refreshed copy to {save_path}",
                    flush=True,
                )

        found_test_id = test_id is None
        for row_number in range(2, worksheet.max_row + 1):
            current_test_id = str(
                worksheet.cell(row_number, headers["Test ID"]).value or ""
            ).strip()
            if test_id and current_test_id != test_id:
                continue
            found_test_id = True

            answer = str(read(row_number, "Generated Answer") or "").strip()
            ground_truth = str(
                read(row_number, "Expected Answer/Ground truth") or ""
            ).strip()
            contexts = _decode_contexts(read(row_number, "Evaluation Contexts"))
            expected_routes = _expected_routes(read(row_number, "chosen routes"))
            document_contexts = (
                _decode_contexts(read(row_number, "Retrieved Chunks"))
                if "documents" in expected_routes
                else []
            )
            generated_sql = str(
                read(row_number, "Generated sql query") or ""
            ).strip()
            contexts_populated = False
            if not contexts:
                if "sql" in expected_routes and generated_sql:
                    sql_context = f"Generated SQL: {generated_sql}"
                    if answer:
                        sql_context += f"\nGenerated answer: {answer}"
                    contexts = [sql_context]
                if "documents" in expected_routes:
                    if document_contexts:
                        contexts.extend(document_contexts)
                    elif ground_truth:
                        contexts.append(ground_truth)
                if contexts:
                    worksheet.cell(
                        row_number,
                        headers["Evaluation Contexts"],
                        json.dumps(contexts, ensure_ascii=True),
                    )
                    contexts_populated = True

            metric_names = ["faithfulness", "answer_relevancy"]
            if document_contexts:
                metric_names.extend(["context_precision", "context_recall"])

            if skip_scored and all(
                _as_score(read(row_number, field)) is not None
                for field, metric_name in RAGAS_FIELDS.items()
                if metric_name in metric_names
            ):
                if contexts_populated:
                    save_workbook()
                skipped += 1
                continue

            errors = _clean_ragas_errors(read(row_number, "Evaluation Error"))
            scores = {}
            should_score = bool(answer and ground_truth and contexts)
            if should_score:
                try:
                    scores = _run_ragas_metrics(
                        str(read(row_number, "Question") or ""),
                        answer,
                        contexts,
                        ground_truth,
                        document_contexts,
                    )
                except Exception as exc:
                    errors.append(
                        "Ragas evaluation failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
            else:
                skipped += 1

            for field, metric_name in RAGAS_FIELDS.items():
                if should_score and metric_name in metric_names:
                    value = scores.get(metric_name)
                    worksheet.cell(row_number, headers[field], value)
                    if value is None:
                        errors.append(
                            f"Ragas did not return {metric_name}"
                        )
                else:
                    worksheet.cell(row_number, headers[field], "")

            worksheet.cell(
                row_number,
                headers["Evaluation Error"],
                "; ".join(dict.fromkeys(errors)),
            )
            save_workbook()
            if should_score:
                scored += 1
                print(
                    f"Refreshed Ragas metrics for {current_test_id}",
                    flush=True,
                )

        if not found_test_id:
            raise ValueError(f"Test ID not found in report: {test_id}")
        if test_id is None:
            print(
                f"Ragas refresh complete: scored={scored}, skipped={skipped}",
                flush=True,
            )
    finally:
        workbook.close()
    return scored, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh Ragas scores from saved evaluation evidence."
    )
    parser.add_argument(
        "--test-id",
        help="Refresh only one case, for example TC-012.",
    )
    parser.add_argument(
        "--skip-scored",
        action="store_true",
        help="Skip cases that already have all applicable Ragas scores.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=EVAL_REPORT_PATH,
        help="Path to the existing evaluation report workbook.",
    )
    args = parser.parse_args()
    refresh_report(
        report_path=args.report,
        test_id=args.test_id,
        skip_scored=args.skip_scored,
    )


if __name__ == "__main__":
    main()
