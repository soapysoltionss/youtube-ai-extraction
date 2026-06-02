"""
Main pipeline entry point.
Orchestrates extraction → chunking → embedding → summarisation for YouTube videos.
"""

import os
import argparse
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.extractor import (
    extract_video_id,
    get_video_metadata,
    get_transcript,
    extract_plain_text,
)
from src.chunker import chunk_text, chunk_transcript_by_time
from src.vector_store import build_vector_store, load_vector_store, similarity_search
from src.summariser import summarise_transcript

load_dotenv()

DEFAULT_DB = "./chroma_db"
DEFAULT_COLLECTION = "youtube_rag"


def process_video(
    url: str,
    persist_directory: str = DEFAULT_DB,
    collection_name: str = DEFAULT_COLLECTION,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    summarise: bool = True,
) -> dict:
    """
    Full pipeline: extract → chunk → embed → (optionally) summarise.

    Args:
        url: YouTube video URL.
        persist_directory: ChromaDB storage directory.
        collection_name: ChromaDB collection name.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlapping characters between chunks.
        summarise: Whether to generate an LLM summary.

    Returns:
        Dictionary with metadata, summary, and number of chunks stored.
    """
    print(f"\n🎬  Processing: {url}")

    # 1. Extract metadata
    print("📋  Fetching metadata...")
    metadata = get_video_metadata(url)
    print(f"    Title   : {metadata['title']}")
    print(f"    Channel : {metadata['channel']}")
    print(f"    Duration: {metadata['duration']}s")

    # 2. Extract transcript
    print("📝  Fetching transcript...")
    video_id = extract_video_id(url) or metadata["video_id"]
    transcript = get_transcript(video_id)
    if not transcript:
        print("⚠️  No transcript available. Falling back to description.")
        plain_text = metadata.get("description", "")
    else:
        plain_text = extract_plain_text(transcript)
        print(f"    Words   : {len(plain_text.split())}")

    # 3. Chunk text
    print("✂️   Chunking text...")
    chunks = chunk_text(plain_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"    Chunks  : {len(chunks)}")

    # 4. Summarise (optional)
    summary = ""
    if summarise and plain_text:
        print("🤖  Summarising transcript...")
        summary = summarise_transcript(plain_text)
        print(f"\n--- SUMMARY ---\n{summary}\n---------------\n")

    # 5. Build / update vector store
    print("💾  Embedding and storing chunks...")
    vector_store = build_vector_store(
        chunks=chunks,
        metadata=metadata,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    print(f"    Stored {len(chunks)} chunks in '{persist_directory}'.")

    return {
        "metadata": metadata,
        "summary": summary,
        "chunks_stored": len(chunks),
        "vector_store": vector_store,
    }


def query_videos(
    query: str,
    persist_directory: str = DEFAULT_DB,
    collection_name: str = DEFAULT_COLLECTION,
    k: int = 5,
) -> None:
    """Search the vector store and print top results."""
    print(f"\n🔍  Query: {query}\n")
    vector_store = load_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    results = similarity_search(vector_store, query, k=k)
    for i, doc in enumerate(results, 1):
        meta = doc.metadata
        print(f"[{i}] {meta.get('title', 'Unknown')} (chunk {meta.get('chunk_index', '?')})")
        print(f"    URL    : {meta.get('url', '')}")
        print(f"    Excerpt: {doc.page_content[:300]}...\n")


_ASK_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant that answers questions based strictly on the provided \
YouTube video transcript excerpts.

If the answer is not contained in the excerpts, say "I couldn't find that in the video."

--- TRANSCRIPT EXCERPTS ---
{context}
--- END EXCERPTS ---

Question: {question}

Answer:"""
)


def ask(
    question: str,
    persist_directory: str = DEFAULT_DB,
    collection_name: str = DEFAULT_COLLECTION,
    k: int = 6,
    model: str = "gemini-2.5-flash",
) -> None:
    """Retrieve relevant chunks and generate a grounded answer with Gemini."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in your .env file.")

    print(f"\n❓  Question: {question}\n")

    # 1. Retrieve relevant chunks
    vector_store = load_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    results = similarity_search(vector_store, question, k=k)

    if not results:
        print("No relevant content found in the video store.")
        return

    # Show sources
    seen = set()
    print("📎  Sources:")
    for doc in results:
        meta = doc.metadata
        key = (meta.get("title", ""), meta.get("url", ""))
        if key not in seen:
            print(f"    • {meta.get('title', 'Unknown')}  {meta.get('url', '')}")
            seen.add(key)

    # 2. Build context from chunks
    context = "\n\n---\n\n".join(
        f"[{doc.metadata.get('title', '')} — chunk {doc.metadata.get('chunk_index', '')}]\n{doc.page_content}"
        for doc in results
    )

    # 3. Ask Gemini
    llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)
    chain = _ASK_PROMPT | llm | StrOutputParser()

    print("\n💬  Answer:\n")
    answer = chain.invoke({"context": context, "question": question})
    print(answer)


def main():
    parser = argparse.ArgumentParser(
        description="YouTube AI Extraction & RAG Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- process sub-command ---
    process_parser = subparsers.add_parser("process", help="Process a YouTube video")
    process_parser.add_argument("url", help="YouTube video URL")
    process_parser.add_argument(
        "--db", default=DEFAULT_DB, help="ChromaDB persist directory"
    )
    process_parser.add_argument(
        "--collection", default=DEFAULT_COLLECTION, help="ChromaDB collection name"
    )
    process_parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Chunk size in characters"
    )
    process_parser.add_argument(
        "--chunk-overlap", type=int, default=200, help="Chunk overlap in characters"
    )
    process_parser.add_argument(
        "--no-summary", action="store_true", help="Skip LLM summarisation"
    )

    # --- query sub-command ---
    query_parser = subparsers.add_parser(
        "query", help="Query the RAG vector store (returns raw chunks)"
    )
    query_parser.add_argument("query", help="Search query")
    query_parser.add_argument(
        "--db", default=DEFAULT_DB, help="ChromaDB persist directory"
    )
    query_parser.add_argument(
        "--collection", default=DEFAULT_COLLECTION, help="ChromaDB collection name"
    )
    query_parser.add_argument(
        "--k", type=int, default=5, help="Number of results to return"
    )

    # --- ask sub-command ---
    ask_parser = subparsers.add_parser(
        "ask", help="Ask a question and get an AI answer grounded in the videos"
    )
    ask_parser.add_argument("question", help="Your question")
    ask_parser.add_argument(
        "--db", default=DEFAULT_DB, help="ChromaDB persist directory"
    )
    ask_parser.add_argument(
        "--collection", default=DEFAULT_COLLECTION, help="ChromaDB collection name"
    )
    ask_parser.add_argument(
        "--k", type=int, default=6, help="Number of chunks to retrieve"
    )

    args = parser.parse_args()

    if args.command == "process":
        process_video(
            url=args.url,
            persist_directory=args.db,
            collection_name=args.collection,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            summarise=not args.no_summary,
        )
    elif args.command == "query":
        query_videos(
            query=args.query,
            persist_directory=args.db,
            collection_name=args.collection,
            k=args.k,
        )
    elif args.command == "ask":
        ask(
            question=args.question,
            persist_directory=args.db,
            collection_name=args.collection,
            k=args.k,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
