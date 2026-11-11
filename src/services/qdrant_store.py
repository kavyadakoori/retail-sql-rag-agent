import hashlib
import logging
import socket
import time
import uuid
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

from qdrant_client import QdrantClient, models

from src.config import Config
from src.observability.metrics import QDRANT_LATENCY, QDRANT_OPERATIONS

logger = logging.getLogger(__name__)


def _local_client() -> QdrantClient:
    Config.QDRANT_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.warning(
        "using_embedded_qdrant",
        extra={"storage_path": str(Config.QDRANT_LOCAL_PATH)},
    )
    return QdrantClient(path=str(Config.QDRANT_LOCAL_PATH))


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    if not Config.QDRANT_URL:
        if Config.APP_ENV.casefold() == "development":
            return _local_client()
        raise RuntimeError("QDRANT_URL is required for document storage.")

    parsed_url = urlparse(Config.QDRANT_URL)
    is_localhost = parsed_url.hostname in {
        "localhost",
        "127.0.0.1",
        "::1",
    }
    if Config.APP_ENV.casefold() == "development" and is_localhost:
        port = parsed_url.port or (
            443 if parsed_url.scheme == "https" else 6333
        )
        try:
            with socket.create_connection(
                (parsed_url.hostname, port), timeout=0.25
            ):
                pass
        except OSError:
            logger.warning(
                "local_qdrant_unavailable_using_embedded_store",
                extra={"url": Config.QDRANT_URL},
            )
            return _local_client()

    return QdrantClient(
        url=Config.QDRANT_URL,
        api_key=Config.QDRANT_API_KEY,
        timeout=10,
    )


def stable_point_id(
    source: str,
    chunk_index: int,
    content: str,
    document_id: str | None = None,
    tenant_id: str = "default",
    user_id: str = "system",
) -> str:
    digest = hashlib.sha256(
        f"{tenant_id}:{user_id}:{document_id or source}:"
        f"{source}:{chunk_index}:{content}".encode("utf-8")
    ).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def _scope_filter(
    tenant_id: str,
    user_id: str,
    document_id: str | None = None,
    include_system: bool = True,
) -> models.Filter:
    must = [
        models.FieldCondition(
            key="tenant_id", match=models.MatchValue(value=tenant_id)
        )
    ]
    if document_id is not None:
        must.append(
            models.FieldCondition(
                key="document_id", match=models.MatchValue(value=document_id)
            )
        )
    scope_conditions = [
        models.FieldCondition(
            key="user_id", match=models.MatchValue(value=user_id)
        )
    ]
    if include_system:
        scope_conditions.append(
            models.FieldCondition(
                key="user_id", match=models.MatchValue(value="system")
            )
        )
    return models.Filter(must=must, should=scope_conditions)


def ensure_collection(client: QdrantClient | None = None) -> None:
    client = client or get_qdrant_client()
    if client.collection_exists(Config.QDRANT_COLLECTION):
        return
    client.create_collection(
        collection_name=Config.QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=Config.QDRANT_VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )


def upsert_chunks(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    client: QdrantClient | None = None,
) -> int:
    started = time.perf_counter()
    try:
        client = client or get_qdrant_client()
        ensure_collection(client)
        points = [
            models.PointStruct(
                id=stable_point_id(
                    item["source"],
                    item["chunk_index"],
                    item["content"],
                    item.get("document_id"),
                    item.get("tenant_id", "default"),
                    item.get("user_id", "system"),
                ),
                vector=vector,
                payload={
                    **item,
                    "document_id": item.get("document_id", item["source"]),
                    "document_name": item.get("document_name", item["source"]),
                    "tenant_id": item.get("tenant_id", "default"),
                    "user_id": item.get("user_id", "system"),
                },
            )
            for item, vector in zip(chunks, embeddings, strict=True)
        ]
        if points:
            client.upsert(
                collection_name=Config.QDRANT_COLLECTION,
                points=points,
            )
        QDRANT_OPERATIONS.labels("upsert", "success").inc()
        return len(points)
    except Exception:
        QDRANT_OPERATIONS.labels("upsert", "error").inc()
        raise
    finally:
        QDRANT_LATENCY.labels("upsert").observe(time.perf_counter() - started)


def search_documents(
    query_vector: list[float],
    limit: int = 4,
    client: QdrantClient | None = None,
    tenant_id: str = "default",
    user_id: str = "anonymous",
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    try:
        client = client or get_qdrant_client()
        ensure_collection(client)
        response = client.query_points(
            collection_name=Config.QDRANT_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=_scope_filter(tenant_id, user_id),
        )
        QDRANT_OPERATIONS.labels("search", "success").inc()
        return [point.payload or {} for point in response.points]
    except Exception:
        QDRANT_OPERATIONS.labels("search", "error").inc()
        raise
    finally:
        QDRANT_LATENCY.labels("search").observe(time.perf_counter() - started)


def list_documents(
    tenant_id: str,
    user_id: str,
    client: QdrantClient | None = None,
) -> list[dict[str, Any]]:
    client = client or get_qdrant_client()
    ensure_collection(client)
    offset = None
    documents: dict[str, dict[str, Any]] = {}
    while True:
        points, offset = client.scroll(
            collection_name=Config.QDRANT_COLLECTION,
            scroll_filter=_scope_filter(tenant_id, user_id),
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for point in points:
            payload = point.payload or {}
            document_id = payload.get("document_id")
            if not document_id:
                continue
            document = documents.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "document_name": payload.get("document_name", "unknown"),
                    "source": payload.get("source", "unknown"),
                    "tenant_id": payload.get("tenant_id", tenant_id),
                    "user_id": payload.get("user_id", "system"),
                    "chunk_count": 0,
                },
            )
            document["chunk_count"] += 1
        if offset is None:
            break
    return list(documents.values())


def get_document(
    document_id: str,
    tenant_id: str,
    user_id: str,
    client: QdrantClient | None = None,
) -> dict[str, Any] | None:
    client = client or get_qdrant_client()
    ensure_collection(client)
    points, _ = client.scroll(
        collection_name=Config.QDRANT_COLLECTION,
        scroll_filter=_scope_filter(tenant_id, user_id, document_id),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        return None
    payload = points[0].payload or {}
    return {
        "document_id": payload.get("document_id", document_id),
        "document_name": payload.get("document_name", "unknown"),
        "source": payload.get("source", "unknown"),
        "tenant_id": payload.get("tenant_id", tenant_id),
        "user_id": payload.get("user_id", "system"),
    }


def delete_document(
    document_id: str,
    tenant_id: str,
    user_id: str,
    client: QdrantClient | None = None,
) -> bool:
    client = client or get_qdrant_client()
    ensure_collection(client)
    document = get_document(document_id, tenant_id, user_id, client=client)
    if document is None or document["user_id"] != user_id:
        return False
    client.delete(
        collection_name=Config.QDRANT_COLLECTION,
        points_selector=models.FilterSelector(
            filter=_scope_filter(
                tenant_id, user_id, document_id, include_system=False
            )
        ),
        wait=True,
    )
    QDRANT_OPERATIONS.labels("delete", "success").inc()
    return True
