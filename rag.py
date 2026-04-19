"""
Personal RAG Assistant with Memory — CLI interface.

Loads documents from the docs/ folder, indexes them in a local ChromaDB
vector store, and provides a conversational interface backed by an
Anthropic Claude model and a sliding-window conversation memory.
"""

import os

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter


DOCS_PATH = "./docs"
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o"
COSMO_API_BASE = "https://ai.cosmoconsult.com/api/v1"


def load_documents():
    """Load all PDFs, plain-text, and Markdown files from DOCS_PATH."""
    if not os.path.exists(DOCS_PATH):
        os.makedirs(DOCS_PATH)
        print("Created docs/ folder. Add your PDFs and text files there.")
        return []

    loaders = []
    for filename in os.listdir(DOCS_PATH):
        filepath = os.path.join(DOCS_PATH, filename)
        if filename.endswith(".pdf"):
            loaders.append(PyPDFLoader(filepath))
        elif filename.endswith((".txt", ".md")):
            loaders.append(TextLoader(filepath, encoding="utf-8"))

    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    return documents


def split_documents(documents):
    """Split documents into overlapping chunks for indexing."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(documents)


def create_vectorstore(chunks):
    """Embed chunks and persist them in a local ChromaDB collection."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH,
    )
    return vectorstore


def build_chain(vectorstore):
    """Build a ConversationalRetrievalChain with sliding-window memory."""
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=os.environ.get("COSMO_API_KEY"),
        openai_api_base=COSMO_API_BASE,
        temperature=0.3,
    )

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10,  # remember last 10 exchanges
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )

    return chain


def format_sources(source_docs):
    """Return a compact citation string from retrieved source documents."""
    seen = set()
    citations = []
    for doc in source_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        label = os.path.basename(source)
        if page is not None:
            label = f"{label} (p. {page + 1})"
        if label not in seen:
            seen.add(label)
            citations.append(label)
    return ", ".join(citations) if citations else None


def main():
    print("Loading documents...")
    documents = load_documents()

    if not documents:
        print("No documents found. Add files to the docs/ folder and try again.")
        return

    print(f"Loaded {len(documents)} document page(s)")

    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s)")

    vectorstore = create_vectorstore(chunks)
    print("Vector store created")

    chain = build_chain(vectorstore)
    print("\nRAG Assistant ready. Type your questions (type 'quit' to exit):\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        response = chain.invoke({"question": question})
        print(f"\nAssistant: {response['answer']}")

        sources = format_sources(response.get("source_documents", []))
        if sources:
            print(f"Sources: {sources}")

        print()


if __name__ == "__main__":
    main()
