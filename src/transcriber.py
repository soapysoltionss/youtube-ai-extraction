"""
Audio transcriber using faster-whisper.
Used as a fallback when no YouTube transcript is available (e.g. live stream VODs,
gameplay footage with commentary but no captions).
"""

import os
import tempfile
import yt_dlp
from faster_whisper import WhisperModel
from typing import Optional


def download_audio(url: str, output_dir: Optional[str] = None) -> str:
    """
    Download the audio track from a YouTube video using yt-dlp.

    Args:
        url: YouTube video or stream URL.
        output_dir: Directory to save the audio file. Defaults to a temp directory.

    Returns:
        Absolute path to the downloaded .mp3 file.
    """
    out_dir = output_dir or tempfile.mkdtemp()
    out_template = os.path.join(out_dir, "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "96",  # Lower quality = faster + smaller file
            }
        ],
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id", "audio")

    audio_path = os.path.join(out_dir, f"{video_id}.mp3")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found after download: {audio_path}")
    return audio_path


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    language: str = "en",
) -> list[dict]:
    """
    Transcribe an audio file using faster-whisper (runs locally, no API needed).

    Model sizes by speed / accuracy tradeoff:
        tiny   → fastest, lowest accuracy  (~75MB)
        base   → good balance              (~145MB)  ← default
        small  → better accuracy           (~460MB)
        medium → high accuracy             (~1.5GB)

    Args:
        audio_path: Path to the .mp3 (or any audio) file.
        model_size: Whisper model size to use.
        language: Language code for transcription (e.g. 'en').

    Returns:
        List of dicts with 'text', 'start', 'duration' — same format as YouTube transcripts.
    """
    print(f"    Loading Whisper '{model_size}' model...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print("    Transcribing audio (this may take a few minutes)...")
    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,  # Skip silent sections — great for streams with dead air
    )

    transcript = []
    for segment in segments:
        transcript.append(
            {
                "text": segment.text.strip(),
                "start": segment.start,
                "duration": segment.end - segment.start,
            }
        )

    print(f"    Detected language: {info.language} (confidence: {info.language_probability:.0%})")
    return transcript


def transcribe_from_url(
    url: str,
    model_size: str = "base",
    language: str = "en",
    keep_audio: bool = False,
) -> list[dict]:
    """
    Download audio from a YouTube URL and transcribe it end-to-end.

    Args:
        url: YouTube video or stream VOD URL.
        model_size: Whisper model size.
        language: Language code.
        keep_audio: If True, keep the downloaded .mp3 file (for debugging).

    Returns:
        List of transcript dicts with 'text', 'start', 'duration'.
    """
    print("    Downloading audio for Whisper transcription...")
    audio_path = download_audio(url)

    try:
        transcript = transcribe_audio(audio_path, model_size=model_size, language=language)
    finally:
        if not keep_audio and os.path.exists(audio_path):
            os.remove(audio_path)

    return transcript
