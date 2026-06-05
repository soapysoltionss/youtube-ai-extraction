"""
Visual frame analyser using Gemini Vision.
Extracts frames from gameplay videos and describes the visual content —
capturing game state, card names, life totals, board state etc.
that commentary alone may miss.
"""

import os
import base64
import subprocess
import tempfile
import glob
import time
from typing import Optional
import yt_dlp
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


_GAMEPLAY_PROMPT = """You are analysing a screenshot from a card game gameplay video.
Describe what you see, focusing on:
- Card names and types visible on the board or in hand
- Game state: life totals, mana/energy/runes available, turn number
- Board state: which creatures/units are in play and their stats
- Any actions, spells, or abilities being resolved
- Phase or step information if visible on screen

Be specific and factual. If card text is too small to read, describe what you can see.
Keep your response concise but information-dense for use in a RAG knowledge base."""


def _find_ffmpeg() -> str:
    """Locate ffmpeg — checks the local .venv/bin first, then system PATH."""
    local = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".venv", "bin", "ffmpeg",
    )
    if os.path.exists(local):
        return local
    return "ffmpeg"  # fall back to system PATH


def download_video_low_res(url: str, output_dir: str) -> str:
    """
    Download a YouTube video at the lowest available resolution.
    Low-res is sufficient to read card names and game state,
    and keeps file sizes small for local processing.
    """
    out_template = os.path.join(output_dir, "video.%(ext)s")
    ydl_opts = {
        "format": "worstvideo[ext=mp4]/worstvideo/worst[ext=mp4]/worst",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        ext = info.get("ext", "mp4")

    video_path = os.path.join(output_dir, f"video.{ext}")
    if not os.path.exists(video_path):
        # yt-dlp may have merged to mkv
        candidates = glob.glob(os.path.join(output_dir, "video.*"))
        if candidates:
            video_path = candidates[0]
        else:
            raise FileNotFoundError("Video file not found after download.")
    return video_path


def extract_frames(
    video_path: str,
    output_dir: str,
    interval_seconds: int = 30,
) -> list[str]:
    """
    Extract one frame every `interval_seconds` from the video using ffmpeg.

    Returns a sorted list of JPEG frame paths.
    """
    frame_pattern = os.path.join(output_dir, "frame_%04d.jpg")
    cmd = [
        _find_ffmpeg(),
        "-i", video_path,
        "-vf", f"fps=1/{interval_seconds}",
        "-q:v", "3",   # JPEG quality 1–31, lower = better
        "-vsync", "vfr",
        frame_pattern,
        "-y",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg frame extraction failed:\n{result.stderr.decode()}"
        )
    return sorted(glob.glob(os.path.join(output_dir, "frame_*.jpg")))


def describe_frame(
    image_path: str,
    api_key: str,
    model: str = "gemini-2.5-flash",
    prompt: str = _GAMEPLAY_PROMPT,
) -> str:
    """Send a single frame to Gemini Vision and return its description."""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)
    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            },
            {"type": "text", "text": prompt},
        ]
    )
    response = llm.invoke([message])
    return response.content


def extract_visual_context(
    url: str,
    api_key: Optional[str] = None,
    interval_seconds: int = 30,
    model: str = "gemini-2.5-flash",
    requests_per_minute: int = 10,
) -> list[dict]:
    """
    Full pipeline: download video → extract frames → describe each frame with Gemini Vision.

    Respects Gemini free-tier rate limits (default: 10 req/min).

    Returns a list of dicts with 'text', 'start', 'duration' —
    the same format as audio transcripts so they can be chunked and stored together.

    Args:
        url: YouTube video URL.
        api_key: Google API key (defaults to GOOGLE_API_KEY env var).
        interval_seconds: How often to sample a frame (default: every 30s).
        model: Gemini model to use for vision.
        requests_per_minute: Rate limit for Gemini API calls.
    """
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GOOGLE_API_KEY is required for visual analysis.")

    delay = 60.0 / requests_per_minute  # seconds between requests

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Download low-res video
        print("    ⬇️  Downloading video (low resolution)...")
        video_path = download_video_low_res(url, tmp_dir)
        size_mb = os.path.getsize(video_path) / 1_000_000
        print(f"    Downloaded: {os.path.basename(video_path)} ({size_mb:.1f} MB)")

        # 2. Extract frames
        print(f"    🎞️  Extracting frames every {interval_seconds}s...")
        frames = extract_frames(video_path, tmp_dir, interval_seconds)
        print(f"    Extracted {len(frames)} frames.")

        if not frames:
            print("    ⚠️  No frames extracted.")
            return []

        # 3. Describe each frame with Gemini Vision
        print(f"    👁️  Analysing frames with Gemini Vision ({len(frames)} frames, ~{len(frames) * delay / 60:.1f} min)...")
        results = []
        for i, frame_path in enumerate(frames):
            timestamp = i * interval_seconds
            minutes = timestamp // 60
            seconds = timestamp % 60
            print(f"    [{i + 1:>3}/{len(frames)}] @ {minutes:02d}:{seconds:02d}", end="\r")

            try:
                description = describe_frame(frame_path, key, model=model)
                results.append(
                    {
                        "text": f"[Visual frame @ {minutes:02d}:{seconds:02d}] {description}",
                        "start": float(timestamp),
                        "duration": float(interval_seconds),
                    }
                )
            except Exception as e:
                print(f"\n    ⚠️  Frame {i + 1} failed: {e}")

            # Respect rate limit — pause between calls
            if i < len(frames) - 1:
                time.sleep(delay)

        print(f"\n    ✅  Analysed {len(results)}/{len(frames)} frames.")
        return results
