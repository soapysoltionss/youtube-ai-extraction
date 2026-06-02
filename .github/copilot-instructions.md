# YouTube AI Extraction — Copilot Instructions

## Project Overview
Python pipeline that extracts YouTube video transcripts and metadata, chunks them, embeds them with OpenAI, and stores them in ChromaDB for RAG (Retrieval Augmented Generation).

## Tech Stack
- **Python 3.11+**
- **youtube-transcript-api** — transcript extraction
- **yt-dlp** — video metadata
- **LangChain** — text splitting, summarisation chains
- **langchain-openai** — OpenAI embeddings & LLM
- **langchain-chroma** — ChromaDB vector store integration
- **ChromaDB** — local vector database
- **python-dotenv** — environment variable management

## Code Conventions
- All modules live under `src/`
- Functions are typed with Python type hints
- Use `DEFAULT_DB` and `DEFAULT_COLLECTION` constants instead of inline string literals
- Summarisation uses the LangChain `refine` chain by default
- Keep extraction, chunking, embedding, and summarisation in separate modules
- Environment variables are loaded from `.env` via `python-dotenv`

## Important Notes
- `OPENAI_API_KEY` must be set in `.env` before running
- ChromaDB is persisted locally in `./chroma_db/` (gitignored)
- The pipeline gracefully falls back to the video description if no transcript is available
