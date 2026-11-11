from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import mysql.connector
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SCHEMA_FILE = BASE_DIR / "database" / "mysql_schema.sql"
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

TABLE_CONFIG = {
    "stores": {
        "file": DATA_DIR / "stores.csv",
        "columns": ["store_id", "store_name", "region", "city", "store_type"],
    },
    "products": {
        "file": DATA_DIR / "products.csv",
        "columns": [
            "product_id",
            "product_name",
            "category",
            "sub_category",
            "base_price",
        ],
    },
    "returns": {
        "file": DATA_DIR / "returns.csv",
        "columns": ["return_id", "order_id", "return_date", "return_reason"],
    },
    "customers": {
        "file": DATA_DIR / "customers.csv",
        "columns": [
            "customer_id",
            "customer_segment",
            "signup_date",
            "preferred_channel",
            "city",
        ],
    },
    "sales_transactions": {
        "file": DATA_DIR / "sales_transactions.csv",
        "columns": [
            "order_id",
            "order_date",
            "store_id",
            "product_id",
            "customer_id",
            "sales_channel",
            "units_sold",
            "unit_price",
            "discount_pct",
            "payment_status",
            "delivery_status",
        ],
    },
}


def get_db_config() -> dict[str, Any]:
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
    }


def read_schema_statements() -> list[str]:
    sql_text = SCHEMA_FILE.read_text(encoding="utf-8")
    statements = [stmt.strip() for stmt in sql_text.split(";") if stmt.strip()]
    return statements


def normalize_value(column_name: str, value: str) -> Any:
    if value is None:
        return None

    value = value.strip()
    if value == "":
        return None

    if column_name.endswith("_date") or column_name in {
        "signup_date",
        "return_date",
        "order_date",
    }:
        return date.fromisoformat(value)

    if column_name in {"units_sold"}:
        return int(value)

    if column_name in {"base_price", "unit_price", "discount_pct"}:
        return Decimal(value)

    return value


def load_table_rows(
    table_name: str,
    columns: list[str],
    csv_path: Path,
) -> list[tuple[Any, ...]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            normalized = tuple(
                normalize_value(column, row.get(column)) for column in columns
            )
            rows.append(normalized)
    return rows


def create_tables(connection: mysql.connector.MySQLConnection) -> None:
    cursor = connection.cursor()
    for statement in read_schema_statements():
        cursor.execute(statement)
    cursor.close()


def load_data() -> None:
    config = get_db_config()
    connection = mysql.connector.connect(**config)
    try:
        create_tables(connection)
        connection.commit()

        for table_name, table_config in TABLE_CONFIG.items():
            columns = table_config["columns"]
            csv_path = table_config["file"]
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")

            rows = load_table_rows(table_name, columns, csv_path)
            if not rows:
                print(f"No rows found in {csv_path.name}")
                continue

            placeholders = ", ".join(["%s"] * len(columns))
            column_sql = ", ".join(columns)
            update_sql = ", ".join(
                f"{column} = VALUES({column})" for column in columns
            )
            insert_sql = (
                f"INSERT INTO {table_name} ({column_sql}) "
                f"VALUES ({placeholders}) "
                f"ON DUPLICATE KEY UPDATE {update_sql}"
            )

            cursor = connection.cursor()
            cursor.executemany(insert_sql, rows)
            cursor.close()
            connection.commit()
            print(f"Loaded {len(rows)} rows into {table_name}")
    finally:
        connection.close()


if __name__ == "__main__":
    load_data()
