import logging
import re
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agents.graph import agent, conversation_config

router = APIRouter(prefix="/api/v1", tags=["chat"])
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    conversation_id: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    message: str = Field(min_length=1, max_length=4000)
    tenant_id: str = Field(default="default", min_length=1, max_length=128)
    user_id: str = Field(default="anonymous", min_length=1, max_length=128)


@router.post("/chat")
def chat(request: ChatRequest):
    conversation_id = request.conversation_id or str(uuid.uuid4())
    try:
        result = agent.invoke(
            {
                "question": request.message,
                "conversation_id": conversation_id,
                "tenant_id": request.tenant_id,
                "user_id": request.user_id,
                "evidence": {},
                "completed_routes": [],
                "pending_routes": [],
                "routes": [],
                "sub_queries": {},
                "active_query": "",
                "plan_initialized": False,
                "iteration_count": 0,
            },
            config=conversation_config(
                conversation_id, request.tenant_id, request.user_id
            ),
        )
    except Exception as exc:
        logger.exception("chat_agent_request_failed")
        raise HTTPException(
            status_code=500,
            detail="The agent could not complete the request.",
        ) from exc
    evidence = result.get("evidence", {})
    document_evidence = str(evidence.get("documents", ""))
    citations = list(dict.fromkeys(re.findall(
        r"\[Source: ([^\]]+)\]", document_evidence
    )))
    return {
        "conversation_id": conversation_id,
        "answer": result.get("answer", "No answer generated."),
        "evidence": evidence,
        "citations": citations,
        "sources_used": list(evidence),
        "status": "completed",
    }
