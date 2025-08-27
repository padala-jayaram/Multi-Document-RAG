import os
# import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
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
st.set_page_config(page_title="Multi-Document RAG with Groq", layout="wide")
st.title("📄 Multi-Document RAG Q&A with Groq Llama 3")

# File uploader
uploaded_files = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if st.button("Process Documents"):
        documents = []

        # Load each PDF
        for uploaded_file in uploaded_files:
            loader = PyPDFLoader(uploaded_file)
            documents.extend(loader.load())

        # Split text
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        text_chunks = text_splitter.split_documents(documents)

        # Embeddings
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Vector store (🚨 in-memory, no persistence)
        vectorstore = Chroma.from_documents(
            documents=text_chunks,
            embedding=embedding,
            persist_directory=None
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # LLM (Groq Llama-3)
        llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        st.success("✅ Documents processed! You can now ask questions.")

        # Query input
        query = st.text_input("Ask a question about the documents:")
        if query:
            if st.button("Submit Query"):
                response = qa_chain({"query": query})
                st.write("### Answer:")
                st.write(response["result"])

                with st.expander("Sources"):
                    for doc in response["source_documents"]:
                        st.write(doc.metadata.get("source", "Unknown"))