from langchain_core.tools import tool

from src.models.document_models import SearchResult
from src.services.ingestion import get_embeddings
from src.services.qdrant_store import search_documents as search_qdrant


@tool
def search_documents(
    query: str, tenant_id: str = "default", user_id: str = "anonymous"
) -> str:
    """Search internal policy and FAQ documents for relevant evidence."""
    results = search_qdrant(
        get_embeddings().embed_query(query),
        tenant_id=tenant_id,
        user_id=user_id,
    )
    search_result = SearchResult(
        contexts=[item.get("content", "") for item in results],
        sources=[
            (
                f"{item.get('source', 'unknown')}, "
                f"page {item['page_number']}"
                if item.get("page_number") is not None
                else item.get("source", "unknown")
            )
            for item in results
        ],
    )
    if not search_result.contexts:
        return "No relevant information found in the internal documents."

    return "\n\n".join(
        f"[Source: {source}]\n{context}"
        for context, source in zip(
            search_result.contexts, search_result.sources, strict=True
        )
    )
