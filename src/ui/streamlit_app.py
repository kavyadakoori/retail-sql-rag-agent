import sys
import re
import uuid
from pathlib import Path

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def invoke_agent(question: str, conversation_id: str) -> dict:
    from src.config import Config

    response = requests.post(
        f"{Config.API_BASE_URL}/api/v1/ask",
        json={"question": question, "conversation_id": conversation_id},
        timeout=120,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        request_id = response.headers.get("x-request-id")
        suffix = f" (request ID: {request_id})" if request_id else ""
        raise RuntimeError(
            f"Agent API error {response.status_code}: " f"{detail}{suffix}"
        )
    return response.json()


def delete_conversation(
    conversation_id: str, tenant_id: str, user_id: str
) -> None:
    from src.config import Config

    response = requests.delete(
        f"{Config.API_BASE_URL}/api/v1/conversations/{conversation_id}",
        params={"tenant_id": tenant_id, "user_id": user_id},
        timeout=10,
    )
    if not response.ok:
        raise RuntimeError(f"Could not clear conversation: {response.text}")


def upload_document(uploaded_file, tenant_id: str, user_id: str) -> dict:
    from src.config import Config

    response = requests.post(
        f"{Config.API_BASE_URL}/api/v1/documents/upload",
        data={"tenant_id": tenant_id, "user_id": user_id},
        files={
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            )
        },
        timeout=120,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        request_id = response.headers.get("x-request-id")
        suffix = f" (request ID: {request_id})" if request_id else ""
        raise RuntimeError(
            f"Document upload failed ({response.status_code}): "
            f"{detail}{suffix}"
        )
    return response.json()


def render_evidence(evidence: dict) -> None:
    sql_evidence = evidence.get("sql")
    if sql_evidence:
        with st.expander("SQL details"):
            st.code(sql_evidence.get("query", ""), language="sql")
            rows = sql_evidence.get("rows", [])
            if rows:
                st.dataframe(rows, use_container_width=True)

    document_evidence = evidence.get("documents")
    if document_evidence:
        citations, chunks = parse_document_evidence(document_evidence)
        with st.expander("Document evidence"):
            if citations:
                st.caption("Sources: " + ", ".join(citations))
            for chunk in chunks:
                st.write(chunk)


def parse_document_evidence(
    document_evidence: str,
) -> tuple[list[str], list[str]]:
    matches = list(
        re.finditer(r"\[Source: ([^\]]+)\]\s*\n?", document_evidence)
    )
    citations: list[str] = []
    chunks: list[str] = []
    for index, match in enumerate(matches):
        citation = match.group(1).strip()
        if citation not in citations:
            citations.append(citation)
        end = matches[index + 1].start() if index + 1 < len(matches) else None
        chunk = document_evidence[match.end() : end].strip()
        if chunk:
            chunks.append(chunk)
    if not matches and document_evidence.strip():
        chunks.append(document_evidence.strip())
    return citations, chunks


def main() -> None:
    st.set_page_config(
        page_title="RetailIQ Analyst",
        page_icon="📊",
        layout="wide",
    )
    st.title("RetailIQ Analyst")
    st.caption("Ask questions about retail data and internal policies.")

    with st.sidebar:
        st.subheader("Add a document")
        uploaded_file = st.file_uploader(
            "Upload a policy or reference file",
            type=["pdf", "docx", "txt", "md", "csv"],
            max_upload_size=15,
            key="document_upload",
        )
        tenant_id = st.text_input("Tenant ID", value="default")
        user_id = st.text_input("User ID", value="anonymous")
        if st.button(
            "Upload and index",
            type="primary",
            disabled=uploaded_file is None,
            use_container_width=True,
            key="upload_index_button",
        ):
            try:
                result = upload_document(uploaded_file, tenant_id, user_id)
                st.success(
                    f"Indexed {result['document_name']} "
                    f"({result['chunk_count']} chunks)."
                )
            except Exception as exc:
                st.error(str(exc))

        st.subheader("Examples")
        examples = [
            "Which stores had the highest sales?",
            "What is the return policy for damaged items?",
            "Show the return rate by category.",
        ]
        for example in examples:
            if st.button(example, use_container_width=True):
                st.session_state.pending_question = example
        if st.button("Clear conversation", use_container_width=True):
            try:
                delete_conversation(
                    st.session_state.conversation_id, tenant_id, user_id
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state.messages = []
                st.session_state.pending_question = ""
                st.session_state.conversation_id = str(uuid.uuid4())
                st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_evidence(message.get("evidence", {}))

    question = st.chat_input("Ask a retail question")
    question = question or st.session_state.pending_question
    st.session_state.pending_question = ""

    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your question..."):
            try:
                result = invoke_agent(
                    question, st.session_state.conversation_id
                )
                answer = result.get("answer", "No answer generated.")
                evidence = result.get("evidence", {})
                st.markdown(answer)
                render_evidence(evidence)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "evidence": evidence,
                    }
                )
            except Exception as exc:
                message = f"Unable to answer the question: {exc}"
                st.error(message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": message, "evidence": {}}
                )


if __name__ == "__main__":
    main()
