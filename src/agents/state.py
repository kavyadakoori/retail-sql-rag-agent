from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    user_query: str
    question: str
    conversation_id: str
    user_id: str
    tenant_id: str
    completed_routes: list[str]
    pending_routes: list[str]
    routes: list[str]
    plan_initialized: bool
    sub_queries: dict[str, str]
    active_query: str
    evidence: dict[str, Any]
    conversation_history: list[dict[str, Any]]
    answer_from_history: bool
    iteration_count: int
    route: str
    answer: str
    requires_multiple_sources: bool
    confidence: float
    reason: str
