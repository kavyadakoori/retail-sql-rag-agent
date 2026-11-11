from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api import main
from src.api.routes import chat as chat_route
from src.api.routes import documents as documents_route
from src.ui.streamlit_app import parse_document_evidence

client = TestClient(main.app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_endpoint_is_exposed():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "retailiq_http_requests_total" in response.text


def test_ask_validates_question_length():
    response = client.post("/api/v1/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_returns_structured_response(monkeypatch):
    captured = {}

    def invoke(state, config):
        captured["state"] = state
        captured["config"] = config
        return {
            "answer": "Store A led sales.",
            "evidence": {"sql": {"analysis": "Store A led sales."}},
        }

    fake_agent = SimpleNamespace(
        invoke=invoke
    )
    monkeypatch.setattr(main, "agent", fake_agent)
    response = client.post(
        "/api/v1/ask",
        json={"question": "Which store led sales?"},
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "Store A led sales."
    assert response.json()["evidence"]["sql"]
    assert response.json()["conversation_id"]
    assert captured["config"]["configurable"]["thread_id"]
    assert captured["state"]["plan_initialized"] is False
    assert response.headers["x-request-id"]


def test_ask_failure_includes_request_id_for_log_lookup(monkeypatch):
    class BrokenAgent:
        def invoke(self, state, config=None):
            raise RuntimeError("upstream unavailable")

    monkeypatch.setattr(main, "agent", BrokenAgent())
    response = client.post(
        "/api/v1/ask",
        json={"question": "Show top 5 product categories by revenue"},
    )

    assert response.status_code == 500
    assert response.headers["x-request-id"] in response.json()["detail"]
    assert "RuntimeError" in response.json()["detail"]


def test_document_evidence_parser_keeps_citations_and_chunk_text():
    citations, chunks = parse_document_evidence(
        "[Source: faq.txt]\nDelivery takes 3-5 business days.\n\n"
        "[Source: faq.txt]\nMetro delivery uses business days."
    )

    assert citations == ["faq.txt"]
    assert chunks == [
        "Delivery takes 3-5 business days.",
        "Metro delivery uses business days.",
    ]


def test_chat_invokes_agent_and_returns_citations(monkeypatch):
    captured = {}

    class FakeAgent:
        def invoke(self, state, config):
            captured.update(state)
            captured["config"] = config
            return {
                "answer": "Returns are allowed for damaged items.",
                "evidence": {
                    "documents": "[Source: return_policy.pdf]\nPolicy text"
                },
            }

    monkeypatch.setattr(chat_route, "agent", FakeAgent())
    response = client.post(
        "/api/v1/chat",
        json={
            "message": "What is the damaged-item return policy?",
            "tenant_id": "tenant-a",
            "user_id": "user-a",
        },
    )

    assert response.status_code == 200
    assert response.json()["citations"] == ["return_policy.pdf"]
    assert captured["tenant_id"] == "tenant-a"
    assert captured["user_id"] == "user-a"
    assert captured["config"]["configurable"]["thread_id"]


def test_clear_conversation_deletes_scoped_memory(monkeypatch):
    from src.api import main

    captured = {}

    def fake_clear(conversation_id, tenant_id, user_id):
        captured.update(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_id=user_id,
        )

    monkeypatch.setattr(main, "clear_conversation", fake_clear)
    response = client.delete(
        "/api/v1/conversations/conversation-1",
        params={"tenant_id": "tenant-a", "user_id": "user-a"},
    )

    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert captured == {
        "conversation_id": "conversation-1",
        "tenant_id": "tenant-a",
        "user_id": "user-a",
    }


def test_document_upload_indexes_supported_file(monkeypatch):
    captured = {}

    class FakeWorkflow:
        def invoke(self, state):
            captured.update(state)
            return {"total_chunks": 3}

    monkeypatch.setattr(documents_route, "ingestion_workflow", FakeWorkflow())
    response = client.post(
        "/api/v1/documents/upload",
        data={"tenant_id": "tenant-a", "user_id": "user-a"},
        files={
            "file": ("policy.md", b"Return policy contents", "text/markdown")
        },
    )

    assert response.status_code == 201
    assert response.json()["status"] == "indexed"
    assert response.json()["chunk_count"] == 3
    assert captured["filename"] == "policy.md"
    assert captured["tenant_id"] == "tenant-a"
    assert captured["content"] == b"Return policy contents"


def test_document_upload_rejects_unsupported_type():
    response = client.post(
        "/api/v1/documents/upload",
        data={"tenant_id": "tenant-a", "user_id": "user-a"},
        files={"file": ("payload.exe", b"not a document")},
    )

    assert response.status_code == 415


def test_document_management_uses_scoped_storage(monkeypatch):
    captured = {}

    def fake_list(tenant_id, user_id):
        captured.update(tenant_id=tenant_id, user_id=user_id)
        return [{"document_id": "doc-1", "document_name": "policy.pdf"}]

    monkeypatch.setattr(documents_route, "list_qdrant_documents", fake_list)
    response = client.get(
        "/api/v1/documents",
        params={"tenant_id": "tenant-a", "user_id": "user-a"},
    )

    assert response.status_code == 200
    assert response.json()["documents"][0]["document_id"] == "doc-1"
    assert captured == {"tenant_id": "tenant-a", "user_id": "user-a"}


def test_document_delete_returns_not_found_for_unowned_document(monkeypatch):
    monkeypatch.setattr(
        documents_route, "delete_qdrant_document", lambda *args: False
    )
    response = client.delete(
        "/api/v1/documents/doc-1",
        params={"tenant_id": "tenant-a", "user_id": "user-a"},
    )

    assert response.status_code == 404


def test_reindex_reports_completed_after_workflow_finishes(monkeypatch):
    class FakeWorkflow:
        def invoke(self, state):
            return {"total_chunks": 7}

    monkeypatch.setattr(main, "ingestion_workflow", FakeWorkflow())
    response = client.post("/api/v1/ingestion/reindex")

    assert response.status_code == 200
    assert response.json() == {"status": "completed", "chunks": 7}
