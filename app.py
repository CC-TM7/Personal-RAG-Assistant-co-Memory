"""
Personal RAG Assistant with Memory — Streamlit web interface.

Provides a browser-based chat UI with:
  • File uploader that accepts PDFs, TXT, and Markdown files
  • Persistent conversation history displayed in the chat window
  • Source citations shown beneath each assistant response
  • Session-level ChromaDB vector store (rebuilt whenever new files are added)
"""

import os
import shutil
import tempfile

import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "claude-sonnet-4-20250514"
CHROMA_PATH = "./chroma_db_app"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_uploaded_file(uploaded_file):
    """Save *uploaded_file* to a temp location and return its LangChain documents."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path, encoding="utf-8")

    docs = loader.load()
    # Replace the temp path with the original filename in metadata
    for doc in docs:
        doc.metadata["source"] = uploaded_file.name
    os.unlink(tmp_path)
    return docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)


def build_chain(vectorstore):
    llm = ChatAnthropic(model=LLM_MODEL, temperature=0.3)
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10,
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )


def format_sources(source_docs):
    seen, citations = set(), []
    for doc in source_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        label = os.path.basename(source)
        if page is not None:
            label = f"{label} (p. {page + 1})"
        if label not in seen:
            seen.add(label)
            citations.append(label)
    return citations


# ── session-state initialisation ─────────────────────────────────────────────

def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []          # {"role": ..., "content": ..., "sources": [...]}
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = set()  # names of already-indexed files


# ── page layout ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Personal RAG Assistant with Memory")

init_state()

# ── sidebar: file upload & status ────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Your Documents")

    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        help="Your key is used only during this session and never stored.",
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    uploaded_files = st.file_uploader(
        "Upload PDFs, TXT, or Markdown files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_files]
        if new_files:
            with st.spinner(f"Indexing {len(new_files)} file(s)…"):
                all_docs = []
                for f in new_files:
                    try:
                        all_docs.extend(load_uploaded_file(f))
                        st.session_state.indexed_files.add(f.name)
                    except Exception as exc:
                        st.warning(f"Could not load {f.name}: {exc}")

                if all_docs:
                    chunks = split_documents(all_docs)
                    if st.session_state.chain is None:
                        vectorstore = build_vectorstore(chunks)
                    else:
                        # Add new chunks to the existing vector store
                        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                        vectorstore = Chroma(
                            persist_directory=CHROMA_PATH,
                            embedding_function=embeddings,
                        )
                        vectorstore.add_documents(chunks)

                    st.session_state.chain = build_chain(vectorstore)
                    st.success(f"Indexed {len(chunks)} chunk(s) from {len(new_files)} file(s)")

    if st.session_state.indexed_files:
        st.subheader("Indexed files")
        for name in sorted(st.session_state.indexed_files):
            st.markdown(f"- 📄 {name}")

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.chain = None
        st.session_state.indexed_files = set()
        # Remove persisted vector store so it is rebuilt fresh next session
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        st.rerun()

# ── main chat area ────────────────────────────────────────────────────────────

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("📎 Sources: " + " · ".join(msg["sources"]))

# Chat input
if prompt := st.chat_input("Ask a question about your documents…"):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("Please enter your Anthropic API key in the sidebar.")
        st.stop()

    if st.session_state.chain is None:
        st.warning("Please upload at least one document before asking questions.")
        st.stop()

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = st.session_state.chain.invoke({"question": prompt})

        answer = response["answer"]
        sources = format_sources(response.get("source_documents", []))

        st.markdown(answer)
        if sources:
            st.caption("📎 Sources: " + " · ".join(sources))

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
