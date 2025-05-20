# NxTechWorks RAG Document Chatbot

A Streamlit-based Retrieval-Augmented Generation (RAG) chatbot that lets you upload documents (PDF, DOCX, XLSX, CSV, SQLite), index them into a local FAISS vector store, and interact via natural language queries. Advanced features include:

---

## 🔍 Vector Retrieval & Reranking
- **Chunk & Embedding**  
  Documents are split into ~1 000-char chunks, embedded with a SentenceTransformer model.  
- **FAISS Index**  
  Embeddings are stored in a local FAISS index for nearest-neighbor search.  
- **Reranking**  
  Top K hits are reranked with a MiniLM model for higher relevance.

---

## 🖨️ OCR Fallback
If PDF text extraction yields no text (e.g., scanned pages), EasyOCR renders pages to images and runs OCR to recover words.

---

## 💡 Suggested Q&A & Summarization
- **Suggested Questions**  
  After indexing, the LLM suggests three common questions about the document. Each appears as an expandable dropdown showing its answer (RAG-driven).  
- **Document Summary**  
  A hierarchical (map-reduce) summarization pipeline: chunk summaries → combine → final global summary.

---

## 🔄 Contextual Memory Q&A
Maintains the last three user–assistant turns. Each new user question is answered using:
1. A system instruction  
2. The most recent 3 Q&A pairs  
3. Retrieved document context  

This ensures follow-ups build on previous answers.

---

## 📜 Session Chat History
Every user query and bot reply is saved to a SQLite `chat_history.db` and displayed in the sidebar.

---

## 🔧 Supported File Types
- **PDF** (with text and scanned-image fallback)  
- **DOCX**  
- **XLS / XLSX**  
- **CSV**  
- **SQLite `.db`** (extracts first 100 rows per table)

---

## 📂 Project Structure

├── .streamlit/
│ └── config.toml # Streamlit server settings (e.g. maxUploadSize)
├── app.py # Main Streamlit application
├── file_utils.py # File-type detection & text extraction
├── ocr_utils.py # EasyOCR fallback for PDFs
├── embedding_utils.py # Chunking + embedding with SentenceTransformer
├── vector_db.py # FAISS index init, upsert, query
├── reranker.py # Reranking with MiniLM
├── llm_utils.py # LLM wrappers: llm_answer, contextual_answer, suggestions, summary
├── chat_history.py # SQLite chat history functions
├── requirements.txt # Python dependencies
├── README.md # This file
└── .env # Your GOOGLE_API_KEY (not committed)


---

## 🚀 How It Works

1. **Startup**  
   - Load `.env` and configure `google.generativeai` with your API key.  
   - Initialize FAISS index file and chat-history DB.

2. **Upload & Index**  
   - User uploads one or more supported files.  
   - For each file:  
     - Read bytes, hash to avoid re-indexing duplicates in the same session.  
     - Detect file type, extract text (or OCR fallback).  
   - Concatenate all text → chunk → embed → batch upsert to FAISS.

3. **Suggested Q&A & Summary**  
   - Immediately after indexing:  
     - **Suggested Q&A:** LLM generates three questions about the doc; each is answered via RAG (vector search → rerank → LLM) and displayed in `st.expander`.  
     - **Summary:** Map-reduce summarizer produces a global summary.

4. **User Interaction**  
   - **Chat Input:** “Ask a question…”  
   - On submit:  
     1. Retrieve top-K chunks, rerank to top 3.  
     2. Load last 3 Q&A from `chat_history.db`.  
     3. Build a unified prompt: system instruction + history + context + new question.  
     4. Call LLM → display answer.  
     5. Save the new question + answer to history.

---

## 🏃‍♂️ Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/nxtechworks/RAG.git
cd RAG

# 2. Create & activate a Python venv
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Set up your environment variables
# Create a file named .env in the project root containing:
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# 5. (Optional) Adjust upload size in .streamlit/config.toml
# maxUploadSize is in MB; default is 300

# 6. Run the app
streamlit run app.py

# 7. Interact!
# - Upload documents, see indexing progress.
# - View suggested Q&A & summary.
# - Ask questions—try follow-ups to test contextual memory.

