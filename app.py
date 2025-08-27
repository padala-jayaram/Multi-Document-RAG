import os
# import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
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

st.set_page_config(page_title="📄 Multi-Doc RAG with Groq + FAISS", layout="wide")
st.title("📄 Multi-Document RAG with Groq + FAISS")
st.write("Upload multiple PDFs, ask questions, and get answers powered by Groq LLM.")

# ----------------------
# File Upload
# ----------------------
uploaded_files = st.file_uploader(
    "Upload your PDF files", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    if st.button("📑 Process Documents"):
        all_docs = []
        for uploaded_file in uploaded_files:
            # Save temp file
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load PDF
            loader = PyPDFLoader(uploaded_file.name)
            docs = loader.load()
            all_docs.extend(docs)

        # ----------------------
        # Text Splitter
        # ----------------------
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        text_chunks = text_splitter.split_documents(all_docs)

        # ----------------------
        # Embeddings + Vectorstore (FAISS)
        # ----------------------
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(text_chunks, embedding)

        # Save in session state for queries
        st.session_state.vstore = vectorstore
        st.success("✅ Documents processed & FAISS vectorstore created!")

# ----------------------
# Question Answering
# ----------------------
if "vstore" in st.session_state:
    query = st.text_input("💡 Ask a question about your documents:")

    if st.button("🔍 Submit Query"):
        retriever = st.session_state.vstore.as_retriever(search_kwargs={"k": 3})

        # Groq LLM
        llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

        # RetrievalQA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        response = qa_chain(query)

        # Show Answer
        st.subheader("📝 Answer")
        st.write(response["result"])

        # Show Sources
        with st.expander("📚 Source Documents"):
            for doc in response["source_documents"]:
                st.markdown(doc.page_content[:500] + "...")