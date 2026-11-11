import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_client import make_asgi_app

from src.api.routes.chat import router as chat_router
from src.api.routes.documents import router as documents_router
from src.api.routes.health import router as health_router
from src.agents.graph import agent, clear_conversation, conversation_config
from src.observability.logging_config import configure_logging, request_id
from src.observability.metrics import HTTP_LATENCY, HTTP_REQUESTS
from src.workflows.ingestion_workflow import ingestion_workflow

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="RetailIQ Agent API", version="1.0.0")
app.mount("/metrics", make_asgi_app())
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(documents_router)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    tenant_id: str = Field(default="default", min_length=1, max_length=128)
    user_id: str = Field(default="anonymous", min_length=1, max_length=128)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    identifier = request.headers.get("x-request-id", str(uuid.uuid4()))
    token = request_id.set(identifier)
    started = time.perf_counter()
    try:
        response = await call_next(request)
        response.headers["x-request-id"] = identifier
        status = str(response.status_code)
        HTTP_REQUESTS.labels(request.method, request.url.path, status).inc()
        HTTP_LATENCY.labels(request.method, request.url.path).observe(
            time.perf_counter() - started
        )
        logger.info(
            "http_request_completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(
                    (time.perf_counter() - started) * 1000, 2
                ),
            },
        )
        return response
    finally:
        request_id.reset(token)


@app.post("/api/v1/ask")
def ask(request: AskRequest) -> dict:
    identifier = request_id.get()
    conversation_id = request.conversation_id or str(uuid.uuid4())
    started = time.perf_counter()
    try:
        result = agent.invoke(
            {
                "question": request.question,
                "conversation_id": conversation_id,
                "tenant_id": request.tenant_id,
                "user_id": request.user_id,
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
            },
            config=conversation_config(
                conversation_id, request.tenant_id, request.user_id
            ),
        )
        logger.info(
            "agent_request_completed",
            extra={
                "duration_ms": round(
                    (time.perf_counter() - started) * 1000, 2
                ),
                "evidence_types": list(result.get("evidence", {}).keys()),
            },
        )
        return {
            "answer": result.get("answer", "No answer generated."),
            "evidence": result.get("evidence", {}),
            "conversation_id": conversation_id,
            "request_id": identifier,
        }
    except Exception as exc:
        logger.exception(
            "agent_request_failed",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=500,
            detail=(
                f"Agent request failed ({type(exc).__name__}). "
                f"Check backend logs using request ID {identifier}."
            ),
        ) from exc


@app.post("/api/v1/ingestion/reindex")
def request_reindex() -> dict[str, object]:
    try:
        result = ingestion_workflow.invoke(
            {"documents": [], "status": "queued"}
        )
    except Exception as exc:  # pragma: no cover - defensive path
        logger.exception(
            "ingestion_workflow_failed",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=500,
            detail="Ingestion workflow failed.",
        ) from exc
    return {"status": "completed", "chunks": result.get("total_chunks", 0)}


@app.delete("/api/v1/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, tenant_id: str, user_id: str):
    clear_conversation(conversation_id, tenant_id, user_id)
    return {"conversation_id": conversation_id, "deleted": True}
