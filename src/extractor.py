"""
YouTube video information extractor.
Fetches transcript and metadata from YouTube videos.
"""

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import yt_dlp
import re
import ssl
import certifi
import requests
from typing import Optional


def extract_video_id(url: str) -> Optional[str]:
    """Extract the YouTube video ID from a URL."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_video_metadata(url: str) -> dict:
    """Fetch video metadata using yt-dlp."""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "nocheckcertificate": True,  # Handle corporate SSL inspection proxies
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            "title": info.get("title", ""),
            "description": info.get("description", ""),
            "channel": info.get("uploader", ""),
            "channel_id": info.get("uploader_id", ""),
            "upload_date": info.get("upload_date", ""),
            "duration": info.get("duration", 0),
            "view_count": info.get("view_count", 0),
            "like_count": info.get("like_count", 0),
            "tags": info.get("tags", []),
            "url": url,
            "video_id": info.get("id", ""),
        }


def get_transcript(video_id: str, languages: list[str] = ["en"]) -> list[dict]:
    """Fetch transcript for a YouTube video."""
    # Use a session that skips SSL verification for corporate proxy environments
    session = requests.Session()
    session.verify = False
    requests.packages.urllib3.disable_warnings()

    try:
        transcript_api = YouTubeTranscriptApi(http_client=session)
        transcript = transcript_api.fetch(video_id, languages=languages)
        return [{"text": s.text, "start": s.start, "duration": s.duration} for s in transcript]
    except TranscriptsDisabled:
        print(f"Transcripts are disabled for video {video_id}.")
        return []
    except NoTranscriptFound:
        print(f"No transcript found for video {video_id} in languages: {languages}.")
        try:
            transcript_list = YouTubeTranscriptApi(http_client=session).list(video_id)
            for t in transcript_list:
                fetched = t.fetch()
                return [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched]
        except Exception as e:
            print(f"Could not retrieve any transcript: {e}")
            return []


def extract_full_text(transcript: list[dict]) -> str:
    """Convert transcript list into a single string with timestamps."""
    if not transcript:
        return ""
    lines = []
    for entry in transcript:
        start = entry["start"]
        minutes = int(start // 60)
        seconds = int(start % 60)
        lines.append(f"[{minutes:02d}:{seconds:02d}] {entry['text']}")
    return "\n".join(lines)


def extract_plain_text(transcript: list[dict]) -> str:
    """Convert transcript list into plain text (no timestamps)."""
    return " ".join(entry["text"] for entry in transcript)
