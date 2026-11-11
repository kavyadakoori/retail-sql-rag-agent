from prometheus_client import Counter, Histogram

HTTP_REQUESTS = Counter(
    "retailiq_http_requests_total",
    "HTTP requests handled by the API.",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "retailiq_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "path"],
)
LLM_REQUESTS = Counter(
    "retailiq_llm_requests_total",
    "LLM requests by model and outcome.",
    ["model", "status"],
)
LLM_TOKENS = Counter(
    "retailiq_llm_tokens_total",
    "LLM tokens consumed by model and token type.",
    ["model", "token_type"],
)
DATABASE_QUERIES = Counter(
    "retailiq_database_queries_total",
    "Database queries by outcome.",
    ["status"],
)
DATABASE_LATENCY = Histogram(
    "retailiq_database_query_duration_seconds",
    "Database query duration in seconds.",
)
QDRANT_OPERATIONS = Counter(
    "retailiq_qdrant_operations_total",
    "Qdrant operations by operation and outcome.",
    ["operation", "status"],
)
QDRANT_LATENCY = Histogram(
    "retailiq_qdrant_operation_duration_seconds",
    "Qdrant operation duration in seconds.",
    ["operation"],
)
