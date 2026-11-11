from pathlib import Path

from src.services import ingestion
from src.workflows.ingestion_workflow import ingest_step


def test_ingestion_loads_chunks_embeds_and_upserts(
    tmp_path: Path, monkeypatch
):
    (tmp_path / "policy.txt").write_text(
        "Return this item within thirty days.", encoding="utf-8"
    )
    captured = {}

    class FakeEmbeddings:
        def embed_documents(self, texts):
            captured["texts"] = texts
            return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(ingestion, "get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr(
        ingestion,
        "upsert_chunks",
        lambda chunks, vectors: captured.update(
            {"chunks": chunks, "vectors": vectors}
        )
        or len(chunks),
    )

    assert ingestion.ingest_documents(tmp_path) == 1
    assert captured["texts"] == ["Return this item within thirty days."]
    assert captured["chunks"][0]["source"] == "policy.txt"
    assert captured["vectors"] == [[0.1, 0.2]]


def test_pdf_extraction_preserves_page_numbers(monkeypatch):
    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakePdfReader:
        def __init__(self, content):
            self.pages = [
                FakePage("Page one policy"),
                FakePage("Page two FAQ"),
            ]

    monkeypatch.setattr(ingestion, "PdfReader", FakePdfReader)
    documents = ingestion.extract_file_documents("guide.pdf", b"%PDF-test")

    page_numbers = [document.metadata["page_number"] for document in documents]
    assert page_numbers == [1, 2]
    assert [document.metadata["source"] for document in documents] == [
        "guide.pdf",
        "guide.pdf",
    ]


def test_uploaded_file_is_indexed_with_owner_metadata(monkeypatch):
    captured = {}

    class FakeEmbeddings:
        def embed_documents(self, texts):
            return [[0.25, 0.75] for _ in texts]

    monkeypatch.setattr(ingestion, "get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr(
        ingestion,
        "upsert_chunks",
        lambda chunks, vectors: captured.update(chunks=chunks, vectors=vectors)
        or len(chunks),
    )

    count = ingestion.ingest_uploaded_file(
        filename="customer-guide.txt",
        content=b"Returns require a receipt.",
        tenant_id="tenant-a",
        user_id="user-a",
        document_id="doc-123",
    )

    assert count == 1
    assert captured["chunks"][0]["document_id"] == "doc-123"
    assert captured["chunks"][0]["tenant_id"] == "tenant-a"
    assert captured["chunks"][0]["user_id"] == "user-a"


def test_ingestion_workflow_indexes_uploaded_file(monkeypatch):
    captured = {}

    def fake_ingest_uploaded_file(**kwargs):
        captured.update(kwargs)
        return 2

    monkeypatch.setattr(
        "src.workflows.ingestion_workflow.ingest_uploaded_file",
        fake_ingest_uploaded_file,
    )
    result = ingest_step(
        {
            "filename": "return-guide.pdf",
            "content": b"file contents",
            "tenant_id": "tenant-a",
            "user_id": "user-a",
            "document_id": "doc-1",
        }
    )

    assert result["status"] == "completed"
    assert result["total_chunks"] == 2
    assert captured["document_id"] == "doc-1"
