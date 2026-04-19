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

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "anthropic/claude-opus-4-7"
COSMO_API_BASE = "https://ai.cosmoconsult.com/api/v1"
CHROMA_PATH = "./chroma_db_app"

AVAILABLE_MODELS = {
    "anthropic/claude-opus-4-7": "Claude Opus 4.7",
    "anthropic/claude-sonnet-4-5": "Claude Sonnet 4.5 (schneller)",
    "openai/gpt-4o": "GPT-4o",
    "openai/gpt-4o-mini": "GPT-4o mini (günstig)",
}


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


def build_chain(vectorstore, model=None):
    llm = ChatOpenAI(
        model=model or LLM_MODEL,
        api_key=os.environ.get("COSMO_API_KEY"),
        openai_api_base=COSMO_API_BASE,
    )
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
        k=10,
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 20},
        ),
        memory=memory,
        return_source_documents=True,
        output_key="answer",
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
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = LLM_MODEL

    # Auto-load existing ChromaDB on startup
    if st.session_state.chain is None and os.path.exists(CHROMA_PATH):
        try:
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
            if vectorstore._collection.count() > 0:
                st.session_state.chain = build_chain(vectorstore, model=st.session_state.selected_model)
        except Exception:
            pass


# ── page layout ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Personal RAG Assistant with Memory")

init_state()

# ── sidebar: file upload & status ────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Your Documents")

    api_key = st.text_input(
        "COSMO API key",
        type="password",
        help="Your key is used only during this session and never stored.",
    )
    if api_key:
        os.environ["COSMO_API_KEY"] = api_key

    model_keys = list(AVAILABLE_MODELS.keys())
    selected_model = st.selectbox(
        "🧠 Modell",
        options=model_keys,
        format_func=lambda x: AVAILABLE_MODELS[x],
        index=model_keys.index(st.session_state.selected_model)
        if st.session_state.selected_model in model_keys else 0,
    )
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        if st.session_state.chain is not None:
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            _vs = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
            st.session_state.chain = build_chain(_vs, model=selected_model)
            st.success(f"Modell gewechselt: {AVAILABLE_MODELS[selected_model]}")

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

                    st.session_state.chain = build_chain(vectorstore, model=st.session_state.selected_model)
                    st.success(f"Indexed {len(chunks)} chunk(s) from {len(new_files)} file(s)")

    # Show all files known from current session uploads
    db_files = set()
    if os.path.exists(CHROMA_PATH):
        try:
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            _vs = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
            results = _vs.get(include=["metadatas"])
            for meta in results.get("metadatas", []):
                src = meta.get("source")
                if src:
                    db_files.add(os.path.basename(src))
        except Exception:
            pass

    all_known_files = db_files | st.session_state.indexed_files
    if all_known_files:
        st.subheader("📚 In der Wissensdatenbank")
        for name in sorted(all_known_files):
            st.markdown(f"- 📄 {name}")

    if st.button("🗄️ Wissensdatenbank löschen", type="secondary"):
        st.session_state["confirm_db_delete"] = True

    if st.session_state.get("confirm_db_delete"):
        st.warning("⚠️ Alle Dokumente aus der Datenbank werden unwiderruflich gelöscht!")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Ja, löschen", type="primary"):
                st.session_state.messages = []
                st.session_state.chain = None
                st.session_state.indexed_files = set()
                st.session_state["confirm_db_delete"] = False
                if os.path.exists(CHROMA_PATH):
                    shutil.rmtree(CHROMA_PATH)
                st.success("Datenbank gelöscht.")
                st.rerun()
        with col2:
            if st.button("❌ Abbrechen"):
                st.session_state["confirm_db_delete"] = False
                st.rerun()

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
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
    if not os.environ.get("COSMO_API_KEY"):
        st.warning("Please enter your COSMO API key in the sidebar.")
        st.stop()

    if st.session_state.chain is None:
        st.warning("Please upload at least one document before asking questions (or enter your API key first to load the existing database).")
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
