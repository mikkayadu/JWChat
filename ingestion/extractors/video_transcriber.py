"""
Video / Audio Ingestion
───────────────────────
Uses OpenAI Whisper (runs locally, free) to transcribe video/audio files.

Supported inputs:
  - Direct video URLs:  https://yoursite.com/videos/lecture.mp4
  - YouTube / Vimeo:   https://youtube.com/watch?v=...
  - Local files:        ./videos/lecture.mp4
  - Any format:         .mp4, .mp3, .wav, .m4a, .webm, .ogg

Nothing is permanently saved to disk — temp files are cleaned up automatically.
"""

import os
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class TranscribedVideo:
    source: str           # file path or URL
    title: str
    transcript: str
    language: str
    duration_seconds: float
    metadata: dict = field(default_factory=dict)


def transcribe_file(file_path: str | Path, model_size: str = "base") -> TranscribedVideo | None:
    """
    Transcribe a local video or audio file using Whisper.

    model_size options (speed vs accuracy):
      "tiny"   — fastest, lower accuracy
      "base"   — good balance (default, recommended to start)
      "small"  — better accuracy
      "medium" — high accuracy (needs more RAM)
      "large"  — best accuracy (needs GPU for reasonable speed)
    """
    # Transparently handle URLs passed here too
    if str(file_path).startswith("http://") or str(file_path).startswith("https://"):
        return transcribe_url(str(file_path), model_size)

    path = Path(file_path)
    if not path.exists():
        logger.error(f"File not found: {path}")
        return None

    logger.info(f"Transcribing: {path.name} (model={model_size})")

    try:
        import whisper
        model = whisper.load_model(model_size)
        result = model.transcribe(str(path), verbose=False)

        transcript = result["text"].strip()
        language = result.get("language", "unknown")
        segments = result.get("segments", [])
        duration = segments[-1]["end"] if segments else 0.0

        logger.success(f"Transcribed {path.name}: {len(transcript)} chars, lang={language}, {duration:.0f}s")

        return TranscribedVideo(
            source=str(path),
            title=path.stem,
            transcript=transcript,
            language=language,
            duration_seconds=duration,
            metadata={
                "file_type": "video",
                "file_name": path.name,
                "whisper_model": model_size,
                "language": language,
                "duration_seconds": round(duration),
            }
        )

    except ImportError:
        logger.error("Whisper not installed. Run: pip install openai-whisper")
        return None
    except Exception as e:
        logger.error(f"Transcription failed for {path}: {e}")
        return None


def transcribe_url(url: str, model_size: str = "base") -> TranscribedVideo | None:
    """
    Transcribe a video/audio from any URL — no manual downloading needed.

    Supports:
      - Direct links:  https://yoursite.com/video.mp4  (streamed, not saved)
      - YouTube:       https://youtube.com/watch?v=...
      - Vimeo:         https://vimeo.com/...
      - 1000+ sites via yt-dlp
    """
    logger.info(f"Processing video URL: {url}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Direct media file? Stream it straight down (faster than yt-dlp)
        if _is_direct_media_url(url):
            audio_path = _download_direct(url, tmp_dir)
            if not audio_path:
                logger.info("Direct download failed, trying yt-dlp as fallback...")
                audio_path = _download_via_ytdlp(url, tmp_dir)
        else:
            # YouTube, Vimeo, etc — use yt-dlp
            audio_path = _download_via_ytdlp(url, tmp_dir)

        if not audio_path:
            logger.error(f"Could not download audio from: {url}")
            return None

        result = transcribe_file(audio_path, model_size)
        if result:
            result.source = url
            result.title = _title_from_url(url)
            result.metadata["source_url"] = url
        return result


def transcribe_url_list(
    urls: list[str],
    model_size: str = "base",
) -> list[TranscribedVideo]:
    """Transcribe a list of video URLs. Returns only successful results."""
    results = []
    for i, url in enumerate(urls, 1):
        logger.info(f"Video {i}/{len(urls)}: {url}")
        result = transcribe_url(url, model_size)
        if result:
            results.append(result)
    return results


def transcribe_directory(
    directory: str | Path,
    model_size: str = "base",
    extensions: tuple = (".mp4", ".mp3", ".wav", ".m4a", ".webm", ".ogg"),
) -> list[TranscribedVideo]:
    """Transcribe all video/audio files in a local directory."""
    directory = Path(directory)
    files = [f for f in directory.rglob("*") if f.suffix.lower() in extensions]
    logger.info(f"Found {len(files)} video/audio files in {directory}")
    results = []
    for f in files:
        result = transcribe_file(f, model_size)
        if result:
            results.append(result)
    return results


# ── Internals ─────────────────────────────────────────────────────────────────

def _is_direct_media_url(url: str) -> bool:
    """True if the URL is a direct link to a media file (not a platform page)."""
    media_extensions = (".mp4", ".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac", ".aac")
    clean = url.split("?")[0].lower()
    return any(clean.endswith(ext) for ext in media_extensions)


def _download_direct(url: str, output_dir: str) -> Path | None:
    """Stream a direct media URL into a temp file. No permanent storage."""
    try:
        import httpx

        file_name = url.split("/")[-1].split("?")[0] or "media.mp4"
        out_path = Path(output_dir) / file_name

        logger.info(f"Streaming: {url}")
        with httpx.stream("GET", url, timeout=120, follow_redirects=True,
                          headers={"User-Agent": "RAG-Bot/1.0"}) as resp:
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=8192):
                    f.write(chunk)

        size_kb = out_path.stat().st_size // 1024
        logger.info(f"Downloaded {file_name} ({size_kb} KB)")
        return out_path

    except Exception as e:
        logger.warning(f"Direct download failed: {e}")
        return None


def _download_via_ytdlp(url: str, output_dir: str) -> Path | None:
    """Use yt-dlp to extract and download audio from platform URLs."""
    try:
        import yt_dlp

        output_template = os.path.join(output_dir, "%(title)s.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }],
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        files = list(Path(output_dir).glob("*.mp3"))
        if files:
            return files[0]

        logger.error("yt-dlp ran but no output file found")
        return None

    except ImportError:
        logger.error("yt-dlp not installed. Run: pip install yt-dlp")
        return None
    except Exception as e:
        logger.error(f"yt-dlp failed for {url}: {e}")
        return None


def _title_from_url(url: str) -> str:
    """Extract a readable title from a URL."""
    name = url.split("/")[-1].split("?")[0]
    return name.rsplit(".", 1)[0].replace("-", " ").replace("_", " ") or url
