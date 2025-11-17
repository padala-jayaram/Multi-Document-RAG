import os
import time
import uuid
import sqlite3
from pathlib import Path
import shutil
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
# from langchain.chains import ConversationalRetrievalChain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

# OCR
from pdf2image import convert_from_path
import pytesseract

# -------------------------
# Config / Secrets
# -------------------------
st.set_page_config(page_title="📄 Multi-Doc RAG Chat (Groq + FAISS + SQLite)", layout="wide")
st.title("📄 Multi-Document RAG Chat with Groq + FAISS (SQLite for chat history)")

# Load .env for local dev fallback
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Optional local tesseract/poppler paths (for Windows)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", None)
POPPLER_PATH = os.getenv("POPPLER_PATH", None)

if not GROQ_API_KEY:
    st.warning("GROQ_API_KEY not set. Set it in Streamlit secrets or your .env as GROQ_API_KEY.")
else:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# -------------------------
# SQLite setup
# -------------------------
DB_PATH = "chat_history.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

c.execute(
    """CREATE TABLE IF NOT EXISTS sessions (
           session_id TEXT PRIMARY KEY,
           created_at INTEGER,
           last_activity INTEGER
       )"""
)
c.execute(
    """CREATE TABLE IF NOT EXISTS messages (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           session_id TEXT,
           role TEXT,
           content TEXT,
           ts INTEGER
       )"""
)
conn.commit()

# -------------------------
# Helpers: DB / session
# -------------------------
def create_session(session_id: str):
    ts = int(time.time())
    c.execute("INSERT OR REPLACE INTO sessions (session_id, created_at, last_activity) VALUES (?, ?, ?)",
              (session_id, ts, ts))
    conn.commit()

def update_last_activity(session_id: str):
    ts = int(time.time())
    c.execute("UPDATE sessions SET last_activity = ? WHERE session_id = ?", (ts, session_id))
    conn.commit()

def get_last_activity(session_id: str):
    r = c.execute("SELECT last_activity FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    return r[0] if r else None

def save_message(session_id: str, role: str, content: str):
    ts = int(time.time())
    c.execute("INSERT INTO messages (session_id, role, content, ts) VALUES (?, ?, ?, ?)",
              (session_id, role, content, ts))
    conn.commit()
    update_last_activity(session_id)

def get_messages(session_id: str):
    rows = c.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id", (session_id,)).fetchall()
    return [{"role": r[0], "content": r[1]} for r in rows]

def clear_session_data(session_id: str):
    c.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    c.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()

    # Clear tmp_uploads folder
    tmp_dir = Path("tmp_uploads")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

# -------------------------
# OCR helper
# -------------------------
def ocr_pdf(pdf_path: str):
    # Windows path override if provided
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    # convert_from_path uses poppler if available
    if POPPLER_PATH:
        pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    else:
        pages = convert_from_path(pdf_path)

    docs = []
    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page)
        docs.append(Document(page_content=text, metadata={"page": i, "source": pdf_path}))
    return docs

# -------------------------
# Session State initialization
# -------------------------
if "vstore" not in st.session_state:
    st.session_state.vstore = None
if "faiss_index_path" not in st.session_state:
    st.session_state.faiss_index_path = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "last_interaction_ts" not in st.session_state:
    st.session_state.last_interaction_ts = None

# Idle timeout seconds (2 minutes)
IDLE_TIMEOUT = 120

# If a session_id exists, check idle timeout on app load
if st.session_state.session_id:
    last_ts = get_last_activity(st.session_state.session_id)
    if last_ts:
        now = int(time.time())
        if now - last_ts > IDLE_TIMEOUT:
            # Expire session
            clear_session_data(st.session_state.session_id)
            st.session_state.session_id = None
            st.session_state.vstore = None
            st.session_state.qa_chain = None
            st.warning("Session expired due to inactivity (>2 minutes). Please upload PDFs again to start a new session.")

# -------------------------
# File Upload & Processing
# -------------------------
uploaded_files = st.file_uploader("Upload your PDF files (multiple allowed)", type=["pdf"], accept_multiple_files=True)

process_col, info_col = st.columns([3,1])

with process_col:
    if uploaded_files:
        if st.button("📑 Process Documents"):
            # create a new session id for this upload
            sid = str(uuid.uuid4())
            st.session_state.session_id = sid
            create_session(sid)

            all_docs = []
            tmp_dir = Path("tmp_uploads")
            tmp_dir.mkdir(parents=True, exist_ok=True)

            for uploaded_file in uploaded_files:
                pdf_path = tmp_dir / uploaded_file.name
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Try normal text extraction
                loader = PyPDFLoader(str(pdf_path))
                docs = loader.load()

                # If no text found → fallback to OCR
                if not any(doc.page_content.strip() for doc in docs):
                    st.warning(f"⚠️ No text detected in {uploaded_file.name}")
                    st.info(f"🔍 Using OCR to extract text from {uploaded_file.name}")
                    docs = ocr_pdf(str(pdf_path))

                # Add metadata source filename (clean)
                for d in docs:
                    if "source" not in d.metadata or not d.metadata["source"]:
                        d.metadata["source"] = uploaded_file.name
                    else:
                        # replace with filename for readability
                        d.metadata["source"] = uploaded_file.name

                all_docs.extend(docs)

            if not all_docs:
                st.error("No text could be extracted from uploaded PDFs.")
            else:
                # Text Splitter
                text_splitter = CharacterTextSplitter(separator="\n",
                                                      chunk_size=1000,
                                                      chunk_overlap=200,
                                                      length_function=len)
                text_chunks = text_splitter.split_documents(all_docs)

                # Embeddings + FAISS
                with st.spinner("Creating embeddings and FAISS vectorstore..."):
                    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    vstore = FAISS.from_documents(text_chunks, embedding)

                    st.session_state.vstore = vstore

                    # create QA chain (conversational)
                    retriever = vstore.as_retriever(search_kwargs={"k": 3})
                    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

                    def create_conversational_rag_chain(llm, retriever):
                        prompt = ChatPromptTemplate.from_template("""
                    You are an assistant. Use the following retrieved context to answer the question.
                    
                    Context:
                    {context}
                    
                    Question:
                    {question}
                    
                    Answer:
                    """)
                    
                        def chain(question):
                            docs = retriever.get_relevant_documents(question)
                            context = "\n\n".join([d.page_content for d in docs])
                            formatted = prompt.format(context=context, question=question)
                    
                            return llm.invoke(formatted)
                    
                        return chain

                    # qa_chain = ConversationalRetrievalChain.from_llm(
                    #     llm=llm,
                    #     retriever=retriever,
                    #     return_source_documents=True
                    # )
                    # document_chain = create_stuff_documents_chain(llm)
                    # retrieval_chain = create_retrieval_chain(
                    #     retriever=vectorstore.as_retriever(),
                    #     combine_documents_chain=document_chain
                    # )
                    # qa_chain = retrieval_chain
                    qa_chain = create_conversational_rag_chain(llm, vectorstore.as_retriever())
                    st.session_state.qa_chain = qa_chain

                st.success("✅ Documents processed and conversational chain ready!")
                st.rerun()

with info_col:
    st.caption("Session controls")
    if st.session_state.session_id:
        st.write("Session id:")
        st.code(st.session_state.session_id)
        if st.button("🗑️ Clear history & end session"):
            clear_session_data(st.session_state.session_id)
            st.session_state.session_id = None
            st.session_state.vstore = None
            st.session_state.qa_chain = None
            st.success("Cleared history and ended session.")
            st.rerun()

# -------------------------
# Chat UI (only when ready)
# -------------------------
if st.session_state.qa_chain and st.session_state.vstore and st.session_state.session_id:
    st.markdown("---")
    st.subheader("💬 Chat with your documents")

    # Load past messages from sqlite
    messages = get_messages(st.session_state.session_id)
    for msg in messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask a question about your documents...")
    if user_input:
        # Check whether session timed out since last action (double-check)
        last_ts = get_last_activity(st.session_state.session_id)
        if last_ts and int(time.time()) - last_ts > IDLE_TIMEOUT:
            # Expire session
            clear_session_data(st.session_state.session_id)
            st.session_state.session_id = None
            st.session_state.vstore = None
            st.session_state.qa_chain = None
            st.warning("Session expired due to inactivity (>2 minutes). Please upload PDFs again to start a new session.")
            st.rerun()

        # Save user message
        save_message(st.session_state.session_id, "user", user_input)
        st.chat_message("user").write(user_input)

        # Build chat history for chain: LangChain expects list of (human, assistant) pairs.
        raw_history = get_messages(st.session_state.session_id)
        # Convert messages to pairs: iterate through messages and build alternating pairs
        pairs = []
        human = None
        assistant = None
        temp_pairs = []
        # For ConversationalRetrievalChain we will pass chat_history as list of tuples (human, assistant)
        # Build last N pairs from raw_history:
        pair_list = []
        last_human = None
        for m in raw_history:
            if m["role"] == "user":
                last_human = m["content"]
            else:
                if last_human is not None:
                    pair_list.append((last_human, m["content"]))
                    last_human = None
        chat_history_for_chain = pair_list  # can be empty list

        # Run the chain
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain({"question": user_input, "chat_history": chat_history_for_chain})

        answer = result.get("answer") or result.get("result") or ""
        # Save assistant message
        save_message(st.session_state.session_id, "assistant", answer)
        st.chat_message("assistant").write(answer)

        # Show sources
        with st.expander("📚 Source Documents"):
            for doc in result.get("source_documents", []):
                # display filename + snippet
                src = doc.metadata.get("source", "unknown")
                snippet = doc.page_content[:500].replace("\n", " ")
                st.markdown(f"**{src}** — {snippet}...")

# If no chain ready, ask user to upload
else:
    st.info("Upload PDFs and press **Process Documents** to start a chat session.")

# -------------------------
# On app exit: close DB
# -------------------------
# (Streamlit keeps the process running; explicit close is optional)
# conn.close()
