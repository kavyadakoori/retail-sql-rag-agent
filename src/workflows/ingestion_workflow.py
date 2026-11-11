from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from src.config import DOCS_DIR
from src.services.ingestion import ingest_documents, ingest_uploaded_file


def ingest_step(state: dict[str, Any]) -> dict[str, Any]:
    filename = state.get("filename")
    content = state.get("content")
    if filename is not None and content is not None:
        total_chunks = ingest_uploaded_file(
            filename=filename,
            content=content,
            tenant_id=state["tenant_id"],
            user_id=state["user_id"],
            document_id=state["document_id"],
        )
        return {
            "document_id": state["document_id"],
            "document_name": filename,
            "status": "completed",
            "total_chunks": total_chunks,
        }

    docs_dir = Path(state.get("documents_dir", DOCS_DIR))
    total_chunks = ingest_documents(docs_dir)
    return {
        "documents": state.get("documents", []),
        "status": "completed",
        "total_chunks": total_chunks,
    }


graph = StateGraph(dict)
graph.add_node("ingest", ingest_step)
graph.set_entry_point("ingest")
graph.add_edge("ingest", END)
ingestion_workflow = graph.compile()
