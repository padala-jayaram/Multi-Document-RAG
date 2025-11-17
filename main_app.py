"""
Streamlit app example: Multi-document RAG (modern LangChain API)

This file replaces older `ConversationalRetrievalChain` usage with:
  - create_history_aware_retriever
  - create_retrieval_chain
  - create_stuff_documents_chain

It includes import fallbacks to improve compatibility across LangChain versions.
Adapt paths/keys/VectorStore backend to your repo specifics if needed.
"""

import streamlit as st
from typing import List, Any, Dict

# --- LangChain imports with compatibility fallbacks ---
try:
    # primary modern APIs
    from langchain.chains import create_history_aware_retriever
except Exception:
    # older packaging split sometimes requires different import surface
    from langchain.chains.history_aware_retriever import create_history_aware_retriever  # type: ignore

try:
    # create_retrieval_chain commonly lives here in recent releases
    from langchain.chains.retrieval import create_retrieval_chain
except Exception:
    try:
        from langchain.chains import create_retrieval_chain  # type: ignore
    except Exception:
        create_retrieval_chain = None  # type: ignore

try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
except Exception:
    # fallback name variants
    from langchain.chains.combine_documents.stuff import create_stuff_documents_chain  # type: ignore

# Prompt templates
try:
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
except Exception:
    # alternate location in some distributions
    try:
        from langchain_core.prompts import ChatPromptTemplate  # type: ignore
        from langchain_core.prompts.chat import MessagesPlaceholder  # type: ignore
    except Exception:
        # Last resort: import from langchain.prompts.chat if available
        from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder  # type: ignore

# Chat LLM and embeddings (OpenAI are common; replace if using another provider)
try:
    from langchain.chat_models import ChatOpenAI
except Exception:
    # some setups use community wrappers (keep fallback minimal)
    from langchain_community.chat_models import ChatOpenAI  # type: ignore

try:
    from langchain.embeddings import OpenAIEmbeddings
except Exception:
    from langchain.embeddings.openai import OpenAIEmbeddings  # type: ignore

# Vector store — adapt to your repository (FAISS/Chroma etc.)
try:
    from langchain.vectorstores import FAISS
except Exception:
    # If your project uses Chroma or another driver, switch accordingly.
    from langchain.vectorstores.faiss import FAISS  # type: ignore

# Document type
from langchain.schema import Document

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Multi-Document RAG (modern LangChain)", layout="wide")

st.title("Multi-Document RAG — modern LangChain example")
st.write("This example uses `create_history_aware_retriever` and `create_retrieval_chain`")

# Simple config input
openai_api_key = st.secrets.get("OPENAI_API_KEY", None) or st.text_input(
    "OpenAI API key (or add as Streamlit secret)", type="password"
)
llm_temperature = st.sidebar.slider("LLM temperature", 0.0, 1.0, 0.0)

# File uploader for documents (basic)
uploaded_files = st.file_uploader("Upload text / PDF documents (multiple)", accept_multiple_files=True)

# Chat history stored in session_state
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict[str, Any]] = []

# Vectorstore object stored in session_state for reuse in the session
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

st.sidebar.markdown("## Controls")
if st.sidebar.button("Clear chat history"):
    st.session_state.chat_history = []

# ---------- Helper utilities ----------
def load_documents_from_uploads(files) -> List[Document]:
    docs: List[Document] = []
    for f in files:
        content = None
        try:
            # simple approach: read as text; for pdfs you might use pdfminer / PyPDF2 etc.
            content = f.getvalue().decode("utf-8")
        except Exception:
            try:
                content = f.getvalue().decode("latin-1")
            except Exception:
                content = None
        if not content:
            continue
        docs.append(Document(page_content=content, metadata={"filename": f.name}))
    return docs

@st.cache_resource
def create_embeddings_and_vectorstore(docs: List[Document]):
    # Create embeddings client
    if not openai_api_key:
        st.error("OpenAI API key required for this example.")
        return None

    # NOTE: if you use a different provider, replace OpenAIEmbeddings with that provider's embeddings class
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Build or persist to FAISS index in memory (for production, persist to disk or external store)
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorstore

def build_rag_chain(vectorstore):
    """
    Build the history-aware retriever + retrieval chain (RAG).
    Returns a runnable chain (rag_chain) and the llm (for any other usage).
    """
    llm = ChatOpenAI(temperature=llm_temperature, openai_api_key=openai_api_key)

    # 1) Base retriever from vectorstore
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 2) Prompt that reformulates follow-up questions into standalone questions
    contextualize_system_prompt = (
        "Given a chat history and the latest user question, "
        "reformulate the user's question as a standalone question which can be "
        "understood without the chat history. Do NOT answer the question."
    )
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}")
    ])

    # 3) create history aware retriever (this will call the llm to create a search query if chat_history exists)
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=contextualize_prompt
    )

    # 4) create answering prompt (this must accept a "context" variable)
    qa_system_prompt = (
        "You are an assistant that answers user questions using the provided retrieved context. "
        "If the answer is not contained in the context, say 'I don't know.' Keep answers concise.\n\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}")
    ])

    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt
    )

    # 5) create retrieval chain
    if create_retrieval_chain is None:
        # possible extremely old/new packaging: try to import dynamically
        try:
            from langchain.chains import create_retrieval_chain as _crc
            rag_chain = _crc(
                retriever=history_aware_retriever,
                combine_docs_chain=combine_docs_chain,
            )
        except Exception as e:
            st.error(f"Could not import create_retrieval_chain: {e}")
            return None, llm
    else:
        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=combine_docs_chain,
        )

    return rag_chain, llm

# ---------- Upload handling & vectorstore creation ----------
if uploaded_files:
    with st.spinner("Loading documents and creating vectorstore..."):
        docs = load_documents_from_uploads(uploaded_files)
        if docs:
            st.session_state.vectorstore = create_embeddings_and_vectorstore(docs)
            st.success(f"Loaded {len(docs)} documents into vectorstore.")
        else:
            st.warning("No readable content found in uploaded files.")

# ---------- Main chat UI ----------
user_question = st.text_input("Ask a question about your uploaded documents")

if user_question and st.session_state.vectorstore is None:
    st.info("Upload documents first (or load a prebuilt vectorstore).")

if user_question and st.session_state.vectorstore is not None:
    with st.spinner("Generating answer..."):
        rag_chain, llm = build_rag_chain(st.session_state.vectorstore)
        if rag_chain is None:
            st.error("RAG chain could not be built; check logs.")
        else:
            # prepare chat_history format: many runtimes expect list of messages
            chat_history = st.session_state.chat_history  # pass through as-is
            # invoke the chain: keys are "input" and "chat_history" per the modern API
            try:
                result = rag_chain.invoke({
                    "input": user_question,
                    "chat_history": chat_history
                })
            except TypeError:
                # some versions might require different call signatures; try simple call
                result = rag_chain({"input": user_question, "chat_history": chat_history})

            # result is typically a dict / RunnableResult — extract text if present
            answer = None
            sources = None
            if isinstance(result, dict):
                # common key names: "output_text", "result", or similar
                answer = result.get("output_text") or result.get("text") or result.get("result") or str(result)
                # some chains include source documents under "source_documents" or "documents"
                sources = result.get("source_documents") or result.get("documents") or None
            else:
                answer = str(result)

            # display
            st.markdown("**Answer:**")
            st.write(answer)

            if sources:
                st.markdown("**Source documents (top results):**")
                for i, doc in enumerate(sources[:5]):
                    # doc may be a Document or dict; handle both
                    try:
                        page_content = doc.page_content
                        meta = getattr(doc, "metadata", {})
                    except Exception:
                        page_content = doc.get("page_content", str(doc))
                        meta = doc.get("metadata", {})
                    st.write(f"- Source {i+1}: {meta.get('filename', meta.get('source', 'unknown'))}")
                    # optionally show excerpt
                    st.write(page_content[:300] + ("..." if len(page_content) > 300 else ""))

            # append to chat history — store minimal: (user, assistant)
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

# ---------- End ----------
