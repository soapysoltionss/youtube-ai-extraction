"""
Text chunker for RAG pipelines.
Splits long text into overlapping chunks suitable for embedding.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: List[str] = ["\n\n", "\n", ". ", " ", ""],
) -> List[str]:
    """
    Split text into overlapping chunks for embedding.

    Args:
        text: The full text to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        separators: Characters/strings to use as split points.

    Returns:
        List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    return splitter.split_text(text)


def chunk_transcript_by_time(
    transcript: list[dict], window_seconds: int = 120
) -> List[dict]:
    """
    Group transcript entries into time-windowed chunks.

    Args:
        transcript: List of transcript entries with 'text', 'start', 'duration'.
        window_seconds: Duration (seconds) of each chunk window.

    Returns:
        List of dicts with 'text', 'start_time', 'end_time'.
    """
    if not transcript:
        return []

    chunks = []
    current_chunk: list[dict] = []
    window_start = transcript[0]["start"]

    for entry in transcript:
        if entry["start"] - window_start > window_seconds and current_chunk:
            chunks.append(
                {
                    "text": " ".join(e["text"] for e in current_chunk),
                    "start_time": current_chunk[0]["start"],
                    "end_time": current_chunk[-1]["start"]
                    + current_chunk[-1]["duration"],
                }
            )
            current_chunk = []
            window_start = entry["start"]
        current_chunk.append(entry)

    if current_chunk:
        chunks.append(
            {
                "text": " ".join(e["text"] for e in current_chunk),
                "start_time": current_chunk[0]["start"],
                "end_time": current_chunk[-1]["start"] + current_chunk[-1]["duration"],
            }
        )

    return chunks
