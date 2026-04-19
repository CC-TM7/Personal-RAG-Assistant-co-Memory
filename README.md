# Personal RAG Assistant with Memory

Ein KI-Chatbot, der deine eigenen Dokumente kennt. PDFs, Notizen oder Textdateien hochladen — und direkt im Chat damit arbeiten. Ein gleitendes Konversationsfenster sorgt dafür, dass Folgefragen funktionieren, ohne Kontext zu verlieren.

Der Assistent läuft vollständig lokal (Embeddings, Vektordatenbank), nur die LLM-Anfragen gehen an die [COSMO AI API](https://ai.cosmoconsult.com).

---

## Features

| Feature | Detail |
|---|---|
| Dokumenten-Ingestion | PDF, TXT, Markdown |
| Vektordatenbank | ChromaDB (lokal, kein Account nötig) |
| Embeddings | `all-MiniLM-L6-v2` via `sentence-transformers` (lokal) |
| LLM-Backend | COSMO AI API (`https://ai.cosmoconsult.com/api/v1`) |
| Modell-Auswahl | Wechsel zwischen Claude Opus, Claude Sonnet, GPT-4o, GPT-4o mini |
| Retrieval | MMR (Maximal Marginal Relevance) für diverse Treffer |
| Memory | Letzte 10 Konversations-Exchanges |
| Quellangaben | Jede Antwort zeigt, aus welchem Dokument sie stammt |
| Persistenz | Vektordatenbank bleibt zwischen Sessions erhalten |
| Interfaces | CLI (`rag.py`) **und** Streamlit Web-UI (`app.py`) |

---

## Setup

### 1. Repository klonen

```bash
git clone https://github.com/CC-TM7/Personal-RAG-Assistant-co-Memory.git
cd Personal-RAG-Assistant-co-Memory
```

### 2. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### 3. API Key konfigurieren

Erstelle eine `.env`-Datei im Projektordner:

```bash
COSMO_API_KEY=dein_key_hier
```

Den Key bekommst du über das [COSMO AI Portal](https://ai.cosmoconsult.com).

> Die `.env`-Datei ist in `.gitignore` eingetragen und wird nie ins Repository committed.

---

## Streamlit Web-UI starten

```bash
python -m streamlit run app.py
```

Öffne <http://localhost:8501> im Browser.

**Sidebar-Funktionen:**
- COSMO API Key eingeben (alternativ via `.env`)
- Modell wählen (Claude Opus, Claude Sonnet, GPT-4o, GPT-4o mini)
- Dokumente per Drag-and-Drop hochladen
- Übersicht aller indizierten Dateien
- Wissensdatenbank löschen (mit Bestätigung)
- Konversation zurücksetzen (Datenbank bleibt erhalten)

---

## CLI-Assistent starten

```bash
python rag.py
```

Legt alle Dokumente aus dem `docs/`-Ordner in die Vektordatenbank und startet eine interaktive Chat-Session im Terminal.

```
Loading documents...
Loaded 42 document page(s)
Split into 187 chunk(s)
Vector store created

RAG Assistant ready. Type your questions (type 'quit' to exit):

You: Was sind die wichtigsten Punkte aus dem Dokument?
Assistant: …
Sources: bericht.pdf (p. 3), bericht.pdf (p. 7)

You: quit
Goodbye!
```

---

## Verfügbare Modelle

| Modell-ID | Anzeigename | Empfehlung |
|---|---|---|
| `anthropic/claude-opus-4-7` | Claude Opus 4.7 | Beste Qualität, komplexe Fragen |
| `anthropic/claude-sonnet-4-5` | Claude Sonnet 4.5 | Schneller, guter Allrounder |
| `openai/gpt-4o` | GPT-4o | Strukturierte Antworten |
| `openai/gpt-4o-mini` | GPT-4o mini | Günstig, für einfache Fragen |

---

## Projektstruktur

```
.
├── app.py              # Streamlit Web-UI
├── rag.py              # CLI RAG-Assistent
├── requirements.txt    # Python-Abhängigkeiten
├── .env                # API Key (nicht committed)
├── .streamlit/
│   └── config.toml     # Streamlit-Konfiguration
├── docs/               # Dokumente für CLI (nicht committed)
└── chroma_db_app/      # Persistente Vektordatenbank (nicht committed)
```

---

## Technischer Hintergrund

- **RAG** (Retrieval-Augmented Generation): Dokumente werden in Chunks zerlegt, als Vektoren gespeichert und bei jeder Anfrage semantisch durchsucht. Nur die relevantesten Passagen werden an das LLM übergeben.
- **MMR-Retrieval**: Statt der `k` ähnlichsten Chunks werden diverse Treffer gewählt, die sich inhaltlich weniger überschneiden — bessere Antwortqualität bei breiten Fragen.
- **Conversation Memory**: Die letzten 10 Frage-Antwort-Paare werden dem LLM als Kontext mitgegeben, sodass Folgefragen ohne Wiederholung funktionieren.
