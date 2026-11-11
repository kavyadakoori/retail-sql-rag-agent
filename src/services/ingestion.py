import logging
from io import BytesIO
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docx import Document as WordDocument
from pypdf import PdfReader

from src.config import Config, DOCS_DIR
from src.services.qdrant_store import upsert_chunks

logger = logging.getLogger(__name__)
SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".csv",
}


def get_embeddings() -> OpenAIEmbeddings:
    if not Config.AI_GATEWAY_API_KEY:
        raise RuntimeError(
            "AI_GATEWAY_API_KEY is required for embeddings."
        )
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=Config.AI_GATEWAY_API_KEY,
        base_url=Config.AI_GATEWAY_URL,
    )


def extract_file_documents(filename: str, content: bytes) -> list[Document]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise ValueError(f"Unsupported document type: {suffix or 'unknown'}")
    if not content:
        raise ValueError("The uploaded file is empty.")

    if suffix == ".pdf":
        try:
            reader = PdfReader(BytesIO(content))
            documents = [
                Document(
                    page_content=page.extract_text() or "",
                    metadata={"source": filename, "page_number": page_number},
                )
                for page_number, page in enumerate(reader.pages, start=1)
            ]
        except Exception as exc:
            raise ValueError("The uploaded PDF could not be read.") from exc
        documents = [doc for doc in documents if doc.page_content.strip()]
        if not documents:
            raise ValueError("The PDF contains no extractable text.")
        return documents

    if suffix == ".docx":
        try:
            word_document = WordDocument(BytesIO(content))
            paragraphs = [
                paragraph.text
                for paragraph in word_document.paragraphs
                if paragraph.text.strip()
            ]
            paragraphs.extend(
                " | ".join(cell.text for cell in row.cells)
                for table in word_document.tables
                for row in table.rows
            )
        except Exception as exc:
            raise ValueError(
                "The uploaded Word document could not be read."
            ) from exc
        text = "\n".join(paragraphs)
        if not text.strip():
            raise ValueError("The Word document contains no extractable text.")
        return [Document(page_content=text, metadata={"source": filename})]

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Text documents must use UTF-8 encoding.") from exc
    if not text.strip():
        raise ValueError("The uploaded file contains no text.")
    return [Document(page_content=text, metadata={"source": filename})]


def load_text_documents(documents_dir: Path = DOCS_DIR) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(documents_dir.iterdir()):
        if (
            not path.is_file()
            or path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS
        ):
            continue
        documents.extend(extract_file_documents(path.name, path.read_bytes()))
    return documents


def _index_documents(
    documents: list[Document],
    document_id: str,
    tenant_id: str,
    user_id: str,
    document_name: str,
) -> int:
    if not documents:
        raise ValueError("No extractable document content was provided.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
    split_documents = splitter.split_documents(documents)
    chunks: list[dict[str, Any]] = []
    for index, document in enumerate(split_documents):
        chunk = {
            "content": document.page_content,
            "source": document.metadata.get("source", document_name),
            "document_id": document_id,
            "document_name": document_name,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "chunk_index": index,
        }
        if "page_number" in document.metadata:
            chunk["page_number"] = document.metadata["page_number"]
        chunks.append(chunk)

    vectors = get_embeddings().embed_documents(
        [chunk["content"] for chunk in chunks]
    )
    count = upsert_chunks(chunks, vectors)
    logger.info(
        "documents_indexed",
        extra={"document_count": 1, "chunk_count": count},
    )
    return count


def ingest_uploaded_file(
    filename: str,
    content: bytes,
    tenant_id: str,
    user_id: str,
    document_id: str,
) -> int:
    documents = extract_file_documents(filename, content)
    return _index_documents(
        documents,
        document_id=document_id,
        tenant_id=tenant_id,
        user_id=user_id,
        document_name=Path(filename).name,
    )


def ingest_documents(documents_dir: Path = DOCS_DIR) -> int:
    documents = load_text_documents(documents_dir)
    by_source: dict[str, list[Document]] = {}
    for document in documents:
        by_source.setdefault(document.metadata["source"], []).append(document)
    return sum(
        _index_documents(
            source_documents,
            document_id=f"system:{source}",
            tenant_id="default",
            user_id="system",
            document_name=source,
        )
        for source, source_documents in by_source.items()
    )
