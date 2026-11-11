import re

BLOCKED_KEYWORDS = [
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "truncate",
    "create",
    "replace",
]
DESTRUCTIVE_REQUEST_PATTERNS = (
    (
        r"^\s*(?:please\s+)?(?:delete|remove|erase|drop|truncate|"
        r"update|modify|alter|insert|overwrite)\b"
    ),
    (
        r"^\s*(?:can|could|would)\s+you\s+(?:please\s+)?"
        r"(?:delete|remove|erase|drop|truncate|update|modify|alter|"
        r"insert|overwrite)\b"
    ),
    (
        r"^\s*i\s+(?:want|need)\s+to\s+(?:delete|remove|erase|drop|"
        r"truncate|update|modify|alter|insert|overwrite)\b"
    ),
)


def is_destructive_request(question: str) -> bool:
    """Detect imperative data-mutation requests before SQL generation."""
    return any(
        re.search(pattern, question, re.IGNORECASE)
        for pattern in DESTRUCTIVE_REQUEST_PATTERNS
    )


def validate_sql(sql: str) -> str:
    """Validate that SQL is a single read-only SELECT statement."""
    cleaned_sql = sql.strip().lower()
    if not cleaned_sql.startswith("select"):
        raise ValueError("Only SELECT statements are allowed.")
    if ";" in cleaned_sql.rstrip(";"):
        raise ValueError("Multiple SQL statements are not allowed.")
    for keyword in BLOCKED_KEYWORDS:
        if re.search(rf"\b{keyword}\b", cleaned_sql):
            raise ValueError(f"Blocked SQL keyword detected: {keyword}")
    return sql.strip().rstrip(";")
