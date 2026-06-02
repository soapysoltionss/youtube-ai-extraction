"""
Summariser module.
Uses Google Gemini to generate concise summaries of video transcripts via LCEL.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Optional
import os


_SUMMARY_PROMPT = PromptTemplate.from_template(
    """You are an expert content analyst. Given the following YouTube video transcript excerpt, \
produce a concise, information-dense summary optimised for use in a Retrieval Augmented \
Generation (RAG) system.

Focus on:
- Key topics, concepts, and facts mentioned
- Named entities (people, companies, products, places)
- Important claims, insights, or conclusions
- Actionable information

Transcript:
{text}

Summary:"""
)

_REFINE_PROMPT = PromptTemplate.from_template(
    """Your task is to produce a final, comprehensive summary of a YouTube video transcript.
We have an existing summary up to a certain point:

{existing_summary}

Below is more of the transcript:

{text}

Refine the summary to incorporate new information. If the new information is not useful, \
return the existing summary unchanged.

Final Summary:"""
)


def summarise_transcript(
    transcript_text: str,
    google_api_key: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
) -> str:
    """
    Summarise a full video transcript using a refine strategy with Gemini.

    Args:
        transcript_text: The full transcript as a plain string.
        google_api_key: Google API key (defaults to GOOGLE_API_KEY env var).
        model: The Gemini model to use.
        chunk_size: Characters per chunk when splitting the transcript.
        chunk_overlap: Overlapping characters between consecutive chunks.

    Returns:
        A concise summary string.
    """
    api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API key is required. Set GOOGLE_API_KEY env var.")

    llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)
    parser = StrOutputParser()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(transcript_text)

    if not chunks:
        return ""

    # First chunk — generate initial summary
    initial_chain = _SUMMARY_PROMPT | llm | parser
    summary = initial_chain.invoke({"text": chunks[0]})

    # Subsequent chunks — refine iteratively
    refine_chain = _REFINE_PROMPT | llm | parser
    for chunk in chunks[1:]:
        summary = refine_chain.invoke({"existing_summary": summary, "text": chunk})

    return summary
