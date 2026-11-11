import logging
import mysql.connector
import time
from mysql.connector import Error
from langchain_core.tools import tool

from src.config import Config
from src.observability.metrics import DATABASE_LATENCY, DATABASE_QUERIES
from src.tools.sql_safety import validate_sql

logger = logging.getLogger(__name__)


@tool
def execute_sql(sql: str) -> list[dict]:
    """Validate and execute a read-only SQL query."""
    validated_sql = validate_sql(sql)
    connection = None
    cursor = None
    started = time.perf_counter()
    status = "error"
    try:
        connection = mysql.connector.connect(
            host=Config.MYSQL_HOST,
            port=Config.MYSQL_PORT,
            database=Config.MYSQL_DATABASE,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
        )
        cursor = connection.cursor(dictionary=True)
        cursor.execute(validated_sql)
        rows = cursor.fetchall()
        DATABASE_QUERIES.labels("success").inc()
        status = "success"
        return rows
    except Error as exc:
        DATABASE_QUERIES.labels("error").inc()
        raise RuntimeError(f"Database error: {exc}") from exc
    finally:
        duration = time.perf_counter() - started
        DATABASE_LATENCY.observe(duration)
        logger.info(
            "sql_execution_timing",
            extra={
                "status": status,
                "duration_ms": round(duration * 1000, 2),
            },
        )
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()
