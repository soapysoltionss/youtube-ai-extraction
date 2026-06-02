# YouTube AI Extraction — RAG Pipeline

Extract transcripts and metadata from YouTube videos, chunk them, embed them with OpenAI, store them in ChromaDB, and query them for Retrieval Augmented Generation (RAG).

---

## Architecture

```
YouTube URL
    │
    ▼
src/extractor.py      ← transcript + metadata via youtube-transcript-api & yt-dlp
    │
    ▼
src/chunker.py        ← splits text into overlapping chunks (LangChain splitter)
    │
    ▼
src/vector_store.py   ← embeds chunks (OpenAI) and stores in ChromaDB
    │
    ▼
src/summariser.py     ← optional LLM summary via LangChain refine chain
    │
    ▼
main.py               ← CLI entry point (process / query)
```

---

## Setup

### 1. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

---

## Usage

### Process a video (extract → chunk → embed → summarise)
```bash
python main.py process "https://www.youtube.com/watch?v=VIDEO_ID"
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `./chroma_db` | ChromaDB persistence directory |
| `--collection` | `youtube_rag` | ChromaDB collection name |
| `--chunk-size` | `1000` | Characters per chunk |
| `--chunk-overlap` | `200` | Overlapping characters between chunks |
| `--no-summary` | — | Skip LLM summarisation (faster, cheaper) |

### Query the RAG store
```bash
python main.py query "What does the video say about machine learning?"
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `./chroma_db` | ChromaDB persistence directory |
| `--collection` | `youtube_rag` | ChromaDB collection name |
| `--k` | `5` | Number of results to return |

---

## Project Structure

```
youtube-ai-extraction/
├── main.py                  # CLI entry point
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── .gitignore
└── src/
    ├── __init__.py
    ├── extractor.py         # YouTube transcript & metadata extraction
    ├── chunker.py           # Text chunking utilities
    ├── vector_store.py      # ChromaDB embedding & retrieval
    └── summariser.py        # LLM-based summarisation
```

---

## How it works for RAG

1. **Extract** — Pulls the full transcript and video metadata (title, channel, date, tags).
2. **Chunk** — Splits the transcript into overlapping windows so no context is lost at chunk boundaries.
3. **Embed** — Each chunk is converted to a vector using `text-embedding-ada-002` (OpenAI).
4. **Store** — Vectors are stored in ChromaDB locally with metadata attached to each chunk.
5. **Query** — At query time, the question is embedded and the nearest chunks are retrieved, ready to be injected into an LLM prompt.
6. **Summarise** — Optionally, a `refine` chain generates a comprehensive summary of the full transcript.
