from pathlib import Path
import hashlib
import json
import logging
import sqlite3
import time
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from src.agents.specialists import run_document_specialist, run_sql_specialist
from src.agents.state import AgentState
from src.agents.supervisor import choose_route, finalize
from src.clients.tiger_gateway import TigerGatewayClient

logger = logging.getLogger(__name__)

llm = TigerGatewayClient()
_memory_path = (
    Path(__file__).resolve().parents[2]
    / ".local"
    / "agent_memory.sqlite"
)
_memory_path.parent.mkdir(parents=True, exist_ok=True)
_memory_connection = sqlite3.connect(_memory_path, check_same_thread=False)
_checkpointer = SqliteSaver(_memory_connection)
_checkpointer.setup()


def call_llm(prompt: str) -> str:
    return llm.call_llm([{"role": "user", "content": prompt}])


def _timed_llm_call(prompt: str, operation: str) -> str:
    started = time.perf_counter()
    status = "error"
    try:
        result = call_llm(prompt)
        status = "success"
        return result
    finally:
        logger.info(
            "llm_call_timing",
            extra={
                "operation": operation,
                "status": status,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )


def supervisor_node(state: AgentState) -> dict[str, Any]:
    return choose_route(
        state, lambda prompt: _timed_llm_call(prompt, "supervisor")
    )


def sql_node(state: AgentState) -> dict[str, Any]:
    return run_sql_specialist(
        state, lambda prompt: _timed_llm_call(prompt, "sql_generation")
    )


def finalize_node(state: AgentState) -> dict[str, str]:
    return finalize(
        state, lambda prompt: _timed_llm_call(prompt, "finalization")
    )


graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("sql_specialist", sql_node)
graph.add_node("document_specialist", run_document_specialist)
graph.add_node("finalize", finalize_node)
graph.set_entry_point("supervisor")
graph.add_conditional_edges(
    "supervisor",
    lambda state: state["route"],
    {
        "sql": "sql_specialist",
        "documents": "document_specialist",
        "finish": "finalize",
    },
)
graph.add_edge("sql_specialist", "supervisor")
graph.add_edge("document_specialist", "supervisor")
graph.add_edge("finalize", END)
agent = graph.compile(checkpointer=_checkpointer)


def conversation_config(
    conversation_id: str,
    tenant_id: str = "default",
    user_id: str = "anonymous",
) -> dict[str, dict[str, str]]:
    identity = json.dumps(
        [tenant_id, user_id, conversation_id],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    thread_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return {"configurable": {"thread_id": thread_id}}


def clear_conversation(
    conversation_id: str,
    tenant_id: str = "default",
    user_id: str = "anonymous",
) -> None:
    thread_id = conversation_config(conversation_id, tenant_id, user_id)[
        "configurable"
    ]["thread_id"]
    _checkpointer.delete_thread(thread_id)


def save_graph_mermaid_source(
    output_path: Path | None = None,
) -> Path:
    target = output_path or (
        Path(__file__).resolve().parents[2] / "evidence" / "agent_workflow.mmd"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(agent.get_graph().draw_mermaid(), encoding="utf-8")
    return target


def save_graph_mermaid_image(
    output_path: Path | None = None,
) -> Path:
    target = output_path or (
        Path(__file__).resolve().parents[2] / "evidence" / "agent_workflow.png"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    image_bytes = agent.get_graph().draw_mermaid_png()
    target.write_bytes(image_bytes)
    return target
