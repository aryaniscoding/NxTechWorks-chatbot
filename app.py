import os
import uuid
import hashlib
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv

from file_utils import detect_file_type, get_page_count, get_text
from ocr_utils import ocr_fallback
from embedding_utils import chunk_and_embed
from vector_db import init_vector_table, upsert_vectors, query_vectors
from reranker import rerank_chunks
from llm_utils import llm_answer, generate_questions, generate_summary,contextual_answer
from chat_history import init_history_db, save_message, load_history

# Load environment variables
load_dotenv()

st.set_page_config(page_title="NxTechWorks RAG Document Chatbot")
st.title("📚 NxTechWorks Document Chatbot")

# Initialize databases
init_vector_table()
init_history_db()

# Create or retrieve session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
sid = st.session_state.session_id

# Track indexed files to avoid re-processing
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

# File uploader
uploaded = st.file_uploader(
    "Upload documents (PDF, DOCX, XLSX, CSV, SQLite .db)",
    type=["pdf", "docx", "xls", "xlsx", "csv", "db"],
    accept_multiple_files=True
)

# We'll capture these after upload completes
chunks = []
full_text = ""

if uploaded:
    full_text = ""
    total_pages = 0

    for f in uploaded:
        file_bytes = f.read()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        # Skip file if already indexed
        if file_hash in st.session_state.indexed_files:
            st.info(f"✅ Skipping previously indexed file: `{f.name}`")
            continue

        ftype = detect_file_type(f)
        pages = get_page_count(BytesIO(file_bytes), ftype)
        total_pages += pages

        text = get_text(BytesIO(file_bytes), ftype)
        if not text.strip():
            if ftype == "pdf":
                text = ocr_fallback(BytesIO(file_bytes))
            else:
                st.error(f"Could not extract text from {f.name}. Unsupported or empty.")
                continue  # Skip

        full_text += "\n\n" + text
        st.session_state.indexed_files.add(file_hash)

    full_text = full_text.strip()
    if full_text:
        # Long vs Short indication
        if total_pages > 15:
            st.warning(f"Long file detected: {total_pages} pages")
        else:
            st.success(f"Short file detected: {total_pages} pages")

        # Chunk & embed the entire document
        chunks, embs = chunk_and_embed(full_text)

        if len(chunks) > 0 and len(embs) > 0:
            upsert_vectors(chunks, embs)
            st.info(f"Indexed {len(chunks)} chunks into vector store.")

            # --- Suggested Questions with Answers ---
            try:
                suggestions = generate_questions(full_text)
                if suggestions:
                    st.markdown("### Suggested Q&A:")
                    for idx, q in enumerate(suggestions, 1):
                        # retrieve context and answer for each suggested question
                        hits = query_vectors(q, top_k=10)
                        top_chunks = rerank_chunks(q, hits, top_k=3)
                        answer = llm_answer(q, top_chunks)
                        # show as dropdown
                        with st.expander(f"{idx}. {q}"):
                            st.write(answer)
            except Exception as e:
                st.error(f"Could not generate Q&A suggestions: {e}")

            # --- Document Summary ---
            try:
                st.markdown("## Document Summary")
                summary_text = generate_summary(chunks)
                st.write(summary_text)
            except Exception as e:
                st.error(f"Could not generate summary: {e}")

        else:
            st.error("Failed to generate valid chunks/embeddings. Check document content.")
    # else:
        # st.error("No text could be extracted from the uploaded files.")

# Sidebar: Chat History
st.sidebar.header("Chat History")
for ts, role, msg in load_history(sid):
    color = "#1f3b4d" if role == "user" else "#333333"
    st.sidebar.markdown(
        f"""
        <div style="
            border:1px solid {color};
            background-color:{color};
            color:white;
            padding:8px;
            border-radius:6px;
            margin-bottom:6px;
        ">
          <strong>{role.title()} @ {ts.split('T')[1][:8]}:</strong><br>{msg}
        </div>""",
        unsafe_allow_html=True
    )

# Main: Contextual Query & Response
query = st.text_input("Ask a question about your documents")
if query:
    # 1) Retrieve relevant chunks
    hits = query_vectors(query, top_k=10)
    top_chunks = rerank_chunks(query, hits, top_k=3)

    # 2) Load last three Q&A pairs from history
    history_rows = load_history(sid)  # returns list of (timestamp, role, message)
    qa_pairs = []
    pending_q = None
    for _, role, msg in history_rows:
        if role == "user":
            pending_q = msg
        elif role == "smart_bot" and pending_q:
            qa_pairs.append((pending_q, msg))
            pending_q = None
    recent_history = qa_pairs[-3:]  # up to last 3 turns

    # 3) Get a contextual answer
    answer = contextual_answer(query, top_chunks, recent_history)

    # 4) Display and save
    st.markdown(f"**Answer:** {answer}")
    save_message(sid, "user", query)
    save_message(sid, "smart_bot", answer)
