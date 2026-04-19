# Personal RAG Assistant with Memory

A chatbot that knows **your** documents. Upload PDFs, notes, or any text files and chat with them in natural language. A sliding-window conversation memory means context builds over time — follow-up questions just work.

---

## What it does

| Feature | Detail |
|---|---|
| Document ingestion | PDF, TXT, Markdown |
| Vector store | ChromaDB (local, no account needed) |
| Embeddings | `all-MiniLM-L6-v2` via `sentence-transformers` |
| LLM | Anthropic Claude (`claude-sonnet-4-20250514`) |
| Memory | Last 10 conversation exchanges |
| Source citations | Every answer shows which document it came from |
| Interfaces | CLI (`rag.py`) **and** Streamlit web UI (`app.py`) |

---

## Setup (≈ 15 minutes)

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/CC-TM7/Personal-RAG-Assistant-co-Memory.git
cd Personal-RAG-Assistant-co-Memory
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY=your_key_here   # Windows: set ANTHROPIC_API_KEY=your_key_here
```

### 4. Add your documents

```bash
# Copy PDFs, markdown files, or text files into docs/
cp ~/path/to/notes.pdf docs/
cp ~/path/to/research.md docs/
```

---

## Run the CLI assistant

```bash
python rag.py
```

```
Loading documents...
Loaded 42 document page(s)
Split into 187 chunk(s)
Vector store created

RAG Assistant ready. Type your questions (type 'quit' to exit):

You: What are the main points from the research paper?
Assistant: …
Sources: research.pdf (p. 3), research.pdf (p. 7)

You: How does this connect to what the other document said about X?
Assistant: …

You: quit
Goodbye!
```

---

## Run the Streamlit web interface

```bash
streamlit run app.py
```

Open <http://localhost:8501> in your browser. You can:

1. Paste your Anthropic API key in the sidebar
2. Upload files via drag-and-drop
3. Chat in the main window — sources are shown under each answer
4. Click **Clear conversation** to start fresh

---

## Project structure

```
.
├── rag.py            # CLI RAG assistant
├── app.py            # Streamlit web UI
├── requirements.txt  # Python dependencies
├── docs/             # Put your documents here (not committed)
└── chroma_db/        # Auto-generated vector store (not committed)
```

---

## What you'll learn

- How **RAG** (Retrieval-Augmented Generation) works end to end
- How vector databases store and retrieve information
- How conversation memory works in LLM applications
- How to build something you'll actually use every day
