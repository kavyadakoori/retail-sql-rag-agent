from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    routes: list[Literal["sql", "documents"]] = Field(default_factory=list)
    answer_from_history: bool = False
    sub_queries: dict[str, str] = Field(default_factory=dict)
    reason: str = ""
    requires_multiple_sources: bool = False
    confidence: float = 0.0
    final_answer_required: bool = False
