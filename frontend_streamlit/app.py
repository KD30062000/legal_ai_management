import streamlit as st
import requests
import os
from typing import List, Dict, Any, Optional
from urllib.parse import quote


try:
    API_BASE_URL = st.secrets["API_BASE_URL"]
except Exception:
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api")


def get_company_documents(company_name: str) -> List[Dict[str, Any]]:
    if not company_name:
        return []
    url = f"{API_BASE_URL}/documents/{quote(company_name)}"
    resp = requests.get(url, timeout=30)
    if not resp.ok:
        return []
    data = resp.json()
    return data.get("documents", [])


def get_document_status(document_id: int) -> Optional[Dict[str, Any]]:
    url = f"{API_BASE_URL}/documents/{document_id}/status"
    resp = requests.get(url, timeout=30)
    if not resp.ok:
        return None
    return resp.json()


def get_sessions(company_name: str) -> List[Dict[str, Any]]:
    if not company_name:
        return []
    url = f"{API_BASE_URL}/chat/sessions/{quote(company_name)}"
    resp = requests.get(url, timeout=30)
    if not resp.ok:
        return []
    data = resp.json()
    return data.get("sessions", [])


def get_session_messages(session_id: int) -> List[Dict[str, Any]]:
    url = f"{API_BASE_URL}/chat/sessions/{session_id}/messages"
    resp = requests.get(url, timeout=30)
    if not resp.ok:
        return []
    data = resp.json()
    return data.get("messages", [])


def upload_document(company_name: str, uploaded_file) -> Optional[Dict[str, Any]]:
    url = f"{API_BASE_URL}/upload/direct"
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    data = {"company_name": company_name}
    resp = requests.post(url, files=files, data=data, timeout=120)
    if not resp.ok:
        try:
            st.error(resp.json().get("detail", resp.text))
        except Exception:
            st.error(resp.text)
        return None
    return resp.json()


def send_chat(
    company_name: str,
    message: str,
    specific_document_id: int,
    session_id: Optional[int] = None,
    session_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    url = f"{API_BASE_URL}/chat"
    payload = {
        "company_name": company_name,
        "message": message,
        "session_id": session_id,
        "session_name": session_name,
        "specific_documents": [specific_document_id],
    }
    resp = requests.post(url, json=payload, timeout=120)
    if not resp.ok:
        try:
            st.error(resp.json().get("detail", resp.text))
        except Exception:
            st.error(resp.text)
        return None
    return resp.json()


def find_messages_for_document(company_name: str, document_id: int) -> List[Dict[str, Any]]:
    sessions = get_sessions(company_name)
    all_messages: List[Dict[str, Any]] = []
    for session in sessions:
        messages = get_session_messages(session["id"])
        for msg in messages:
            context_docs = msg.get("context_documents") or []
            if any(str(document_id) == str(d) for d in context_docs):
                msg_copy = dict(msg)
                msg_copy["session_id"] = session["id"]
                msg_copy["session_name"] = session.get("name")
                all_messages.append(msg_copy)
    all_messages.sort(key=lambda m: m.get("timestamp") or "")
    return all_messages


def find_latest_session_for_document(company_name: str, document_id: int) -> Optional[int]:
    sessions = get_sessions(company_name)
    candidate: Optional[Dict[str, Any]] = None
    for session in sessions:
        messages = get_session_messages(session["id"])
        if any(
            any(str(document_id) == str(d) for d in (m.get("context_documents") or []))
            for m in messages
        ):
            if candidate is None or session.get("updated_at", "") > candidate.get("updated_at", ""):
                candidate = session
    return candidate["id"] if candidate else None


def main() -> None:
    st.set_page_config(page_title="Legal Document AI", layout="wide")

    if "company_name" not in st.session_state:
        st.session_state.company_name = ""
    if "selected_document_id" not in st.session_state:
        st.session_state.selected_document_id = None
    if "selected_document_filename" not in st.session_state:
        st.session_state.selected_document_filename = None

    with st.sidebar:
        st.header("Documents")
        company_name = st.text_input("Company name", value=st.session_state.company_name)
        if company_name != st.session_state.company_name:
            st.session_state.company_name = company_name
            st.session_state.selected_document_id = None
            st.session_state.selected_document_filename = None

        documents = get_company_documents(st.session_state.company_name)
        if not documents and st.session_state.company_name:
            st.caption("No documents yet. Upload one.")

        for doc in documents:
            label = f"{doc.get('filename', 'unknown')}"
            status = doc.get("processed", "pending")
            if status != "completed":
                label += f"  [{status}]"
            if st.button(label, key=f"doc_btn_{doc['id']}"):
                st.session_state.selected_document_id = doc["id"]
                st.session_state.selected_document_filename = doc.get("filename")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Upload & Chat")
        if "_upload_file" not in st.session_state:
            st.session_state._upload_file = None
        up = st.file_uploader(
            "Upload a document", type=["pdf", "docx", "txt"], accept_multiple_files=False, key="_uploader"
        )
        st.session_state._upload_file = up
        if st.session_state._upload_file is not None and st.session_state.company_name:
            if st.button("Upload", key="do_upload"):
                with st.spinner("Uploading..."):
                    result = upload_document(st.session_state.company_name, st.session_state._upload_file)
                if result:
                    st.success(f"Uploaded. Document ID: {result.get('document_id')}")
                    st.session_state.selected_document_id = result.get("document_id")
                    st.session_state.selected_document_filename = st.session_state._upload_file.name
                    # Reset uploader to prevent re-uploads on rerun
                    st.session_state._upload_file = None
                    st.rerun()

        if not st.session_state.company_name:
            st.info("Enter a company name to view documents and chat.")
            return

        if st.session_state.selected_document_id is None:
            st.info("Select a document from the left or upload a new one.")
            return

        doc_status = get_document_status(st.session_state.selected_document_id)
        if doc_status is not None:
            status = doc_status.get("processed")
            st.markdown(
                f"**Selected:** {st.session_state.selected_document_filename or ''} "
                f"(ID {st.session_state.selected_document_id}) — Status: `{status}`"
            )
            if status != "completed":
                st.warning("Document is not processed yet. You can still send a message; results may be limited.")

        st.divider()
        st.markdown("**Chat for selected document**")

        history = find_messages_for_document(st.session_state.company_name, st.session_state.selected_document_id)
        for msg in history:
            role = msg.get("role", "user")
            with st.chat_message("user" if role == "user" else "assistant"):
                st.write(msg.get("content", ""))

        prompt = st.chat_input("Ask about this document…")
        if prompt:
            existing_session_id = find_latest_session_for_document(
                st.session_state.company_name, st.session_state.selected_document_id
            )
            session_name = None
            if existing_session_id is None and st.session_state.selected_document_filename:
                session_name = f"Doc {st.session_state.selected_document_id} - {st.session_state.selected_document_filename}"

            with st.spinner("Thinking…"):
                resp = send_chat(
                    company_name=st.session_state.company_name,
                    message=prompt,
                    specific_document_id=st.session_state.selected_document_id,
                    session_id=existing_session_id,
                    session_name=session_name,
                )
            if resp:
                st.session_state["_last_chat_resp"] = resp
                st.rerun()

    with col_right:
        st.subheader("Details")
        if st.session_state.selected_document_id:
            if st.button("Refresh status"):
                st.rerun()
            if doc_status := get_document_status(st.session_state.selected_document_id):
                st.json(doc_status)


if __name__ == "__main__":
    main()


