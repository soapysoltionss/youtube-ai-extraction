"""
Embeddings and vector store manager.
Uses Google Gemini embeddings and ChromaDB for local vector storage.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List, Optional
import os


def build_vector_store(
    chunks: List[str],
    metadata: dict,
    persist_directory: str = "./chroma_db",
    collection_name: str = "youtube_rag",
    google_api_key: Optional[str] = None,
) -> Chroma:
    """
    Embed text chunks and store them in a ChromaDB vector store.

    Args:
        chunks: List of text chunks to embed.
        metadata: Video metadata to attach to each document.
        persist_directory: Directory to persist the ChromaDB database.
        collection_name: Name of the ChromaDB collection.
        google_api_key: Google API key (defaults to GOOGLE_API_KEY env var).

    Returns:
        A Chroma vector store instance.
    """
    api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API key is required. Set GOOGLE_API_KEY env var.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2", google_api_key=api_key
    )

    documents = [
        Document(
            page_content=chunk,
            metadata={
                "video_id": metadata.get("video_id", ""),
                "title": metadata.get("title", ""),
                "channel": metadata.get("channel", ""),
                "url": metadata.get("url", ""),
                "upload_date": metadata.get("upload_date", ""),
                "chunk_index": i,
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    return vector_store


def load_vector_store(
    persist_directory: str = "./chroma_db",
    collection_name: str = "youtube_rag",
    google_api_key: Optional[str] = None,
) -> Chroma:
    """Load an existing ChromaDB vector store."""
    api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API key is required. Set GOOGLE_API_KEY env var.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2", google_api_key=api_key
    )
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )


def similarity_search(
    vector_store: Chroma,
    query: str,
    k: int = 5,
) -> List[Document]:
    """Search the vector store for the most similar chunks to a query."""
    return vector_store.similarity_search(query, k=k)
