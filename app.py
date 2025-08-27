import os
# import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Load environment variables
# First check Streamlit Secrets (Cloud)
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    # Fallback for local dev with .env
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Streamlit UI
st.set_page_config(page_title="Multi-Document RAG QA", layout="wide")
st.title("📄 Multi-Document RAG QA with Groq + LLaMA3")

# -------------------
# Upload PDFs
# -------------------
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Store pipeline in session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if st.button("Submit Documents") and uploaded_files:
    with st.spinner("Processing documents..."):
        # Save uploaded files locally
        os.makedirs("uploaded_docs", exist_ok=True)
        for file in uploaded_files:
            with open(os.path.join("uploaded_docs", file.name), "wb") as f:
                f.write(file.getbuffer())

        # Load and process documents
        loader = DirectoryLoader("uploaded_docs", glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        st.success(f"✅ Loaded {len(documents)} pages from {len(uploaded_files)} PDFs")

        # Split into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = text_splitter.split_documents(documents)
        st.write(f"🔹 Created {len(text_chunks)} chunks")

        # Embeddings + Vectorstore
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # persist_directory = "chroma_db"
        # shutil.rmtree(persist_directory, ignore_errors=True)  # clear old DB

        vectorstore = Chroma.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            persist_directory=None
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # QA Chain
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

# -------------------
# Query Section
# -------------------
if st.session_state.qa_chain:
    query = st.text_input("Ask a question about your documents:")

    if st.button("Submit Query") and query:
        with st.spinner("Getting answer..."):
            response = st.session_state.qa_chain.invoke({"query": query})

        st.write("### Answer:")
        st.write(response["result"])

        # Show sources
        with st.expander("Sources"):
            for doc in response["source_documents"]:
                st.write(doc.metadata["source"])
