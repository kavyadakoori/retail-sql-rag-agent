from uuid import UUID

from types import SimpleNamespace

from qdrant_client import QdrantClient

from src.services import qdrant_store
from src.services.qdrant_store import stable_point_id


def test_point_id_is_deterministic_uuid():
    first = stable_point_id("return_policy.txt", 1, "30 day returns")
    second = stable_point_id("return_policy.txt", 1, "30 day returns")
    assert first == second
    UUID(first)


def test_point_id_is_scoped_to_owner():
    first = stable_point_id(
        "policy.txt", 0, "return policy", "doc-1", "tenant-a", "user-a"
    )
    second = stable_point_id(
        "policy.txt", 0, "return policy", "doc-1", "tenant-b", "user-a"
    )

    assert first != second


def test_list_documents_aggregates_chunks_and_applies_scope():
    class FakeClient:
        def collection_exists(self, collection_name):
            return True

        def scroll(self, **kwargs):
            self.scroll_filter = kwargs["scroll_filter"]
            return [
                SimpleNamespace(
                    payload={
                        "document_id": "doc-1",
                        "document_name": "policy.pdf",
                        "source": "policy.pdf",
                        "tenant_id": "tenant-a",
                        "user_id": "user-a",
                    }
                ),
                SimpleNamespace(
                    payload={
                        "document_id": "doc-1",
                        "document_name": "policy.pdf",
                        "source": "policy.pdf",
                        "tenant_id": "tenant-a",
                        "user_id": "user-a",
                    }
                ),
            ], None

    client = FakeClient()
    documents = qdrant_store.list_documents("tenant-a", "user-a", client)

    assert len(documents) == 1
    assert documents[0]["chunk_count"] == 2
    assert client.scroll_filter.must[0].match.value == "tenant-a"
    scoped_users = {
        condition.match.value for condition in client.scroll_filter.should
    }
    assert scoped_users == {
        "user-a",
        "system",
    }


def test_qdrant_document_lifecycle_is_tenant_scoped(monkeypatch):
    from src.config import Config

    monkeypatch.setattr(Config, "QDRANT_COLLECTION", "document_lifecycle_test")
    monkeypatch.setattr(Config, "QDRANT_VECTOR_SIZE", 2)
    client = QdrantClient(":memory:")
    try:
        qdrant_store.upsert_chunks(
            [
                {
                    "content": "Tenant A return terms",
                    "source": "a.txt",
                    "document_id": "doc-a",
                    "document_name": "a.txt",
                    "tenant_id": "tenant-a",
                    "user_id": "user-a",
                    "chunk_index": 0,
                },
                {
                    "content": "Tenant B return terms",
                    "source": "b.txt",
                    "document_id": "doc-b",
                    "document_name": "b.txt",
                    "tenant_id": "tenant-b",
                    "user_id": "user-b",
                    "chunk_index": 0,
                },
            ],
            [[1.0, 0.0], [1.0, 0.0]],
            client=client,
        )

        results = qdrant_store.search_documents(
            [1.0, 0.0],
            client=client,
            tenant_id="tenant-a",
            user_id="user-a",
        )
        documents = qdrant_store.list_documents(
            "tenant-a", "user-a", client=client
        )

        assert [result["document_id"] for result in results] == ["doc-a"]
        assert [document["document_id"] for document in documents] == ["doc-a"]
        assert qdrant_store.delete_document(
            "doc-a", "tenant-a", "user-a", client=client
        )
        assert (
            qdrant_store.get_document(
                "doc-a", "tenant-a", "user-a", client=client
            )
            is None
        )
        assert (
            qdrant_store.get_document(
                "doc-b", "tenant-a", "user-a", client=client
            )
            is None
        )
    finally:
        client.close()


def test_development_uses_embedded_store_when_local_qdrant_is_down(
    tmp_path, monkeypatch
):
    from src.config import Config

    monkeypatch.setattr(Config, "APP_ENV", "development")
    monkeypatch.setattr(Config, "QDRANT_URL", "http://127.0.0.1:1")
    monkeypatch.setattr(Config, "QDRANT_LOCAL_PATH", tmp_path / "qdrant")
    qdrant_store.get_qdrant_client.cache_clear()
    try:
        client = qdrant_store.get_qdrant_client()
        qdrant_store.ensure_collection(client)

        assert client.collection_exists(Config.QDRANT_COLLECTION)
        assert (tmp_path / "qdrant").exists()
    finally:
        cached_client = qdrant_store.get_qdrant_client.cache_info().currsize
        if cached_client:
            qdrant_store.get_qdrant_client().close()
        qdrant_store.get_qdrant_client.cache_clear()
