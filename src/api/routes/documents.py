import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from src.config import Config
from src.services.ingestion import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
)
from src.services.qdrant_store import (
    delete_document as delete_qdrant_document,
    get_document as get_qdrant_document,
    list_documents as list_qdrant_documents,
)
from src.workflows.ingestion_workflow import ingestion_workflow

router = APIRouter(prefix="/api/v1", tags=["documents"])
logger = logging.getLogger(__name__)


@router.get("/documents")
def list_documents(
    tenant_id: str = Query(min_length=1, max_length=128),
    user_id: str = Query(min_length=1, max_length=128),
):
    try:
        documents = list_qdrant_documents(tenant_id, user_id)
    except Exception as exc:
        logger.exception("document_list_failed")
        raise HTTPException(
            status_code=503, detail="Document storage is unavailable."
        ) from exc
    return {"documents": documents}


@router.get("/documents/{document_id}")
def get_document(
    document_id: str,
    tenant_id: str = Query(min_length=1, max_length=128),
    user_id: str = Query(min_length=1, max_length=128),
):
    try:
        document = get_qdrant_document(document_id, tenant_id, user_id)
    except Exception as exc:
        logger.exception("document_fetch_failed")
        raise HTTPException(
            status_code=503, detail="Document storage is unavailable."
        ) from exc
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    tenant_id: str = Query(min_length=1, max_length=128),
    user_id: str = Query(min_length=1, max_length=128),
):
    try:
        deleted = delete_qdrant_document(document_id, tenant_id, user_id)
    except Exception as exc:
        logger.exception("document_delete_failed")
        raise HTTPException(
            status_code=503, detail="Document storage is unavailable."
        ) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"document_id": document_id, "deleted": True}


@router.post("/documents/upload", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    tenant_id: str = Form(min_length=1, max_length=128),
    user_id: str = Form(min_length=1, max_length=128),
):
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                "Supported file types are PDF, DOCX, TXT, Markdown, and CSV."
            ),
        )
    content = await file.read(Config.MAX_FILE_SIZE_BYTES + 1)
    if len(content) > Config.MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413, detail="Uploaded file is too large."
        )
    document_id = str(uuid.uuid4())
    try:
        result = await run_in_threadpool(
            ingestion_workflow.invoke,
            {
                "filename": filename,
                "content": content,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "document_id": document_id,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("document_upload_failed")
        raise HTTPException(
            status_code=503,
            detail="The document could not be indexed.",
        ) from exc
    return {
        "document_id": document_id,
        "document_name": filename,
        "chunk_count": result.get("total_chunks", 0),
        "status": "indexed",
    }
