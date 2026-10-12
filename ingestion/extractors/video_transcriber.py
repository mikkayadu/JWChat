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


def transcribe_file(
    file_path: str | Path,
    model_size: str = "base",
    language: str | None = None,
) -> TranscribedVideo | None:
    """
    Transcribe a local video or audio file using Whisper.
    ``language`` is the registry code the file is ingested under; files in a
    language Whisper can't transcribe (e.g. Twi) are skipped.

    model_size options (speed vs accuracy):
      "tiny"   — fastest, lower accuracy
      "base"   — good balance (default, recommended to start)
      "small"  — better accuracy
      "medium" — high accuracy (needs more RAM)
      "large"  — best accuracy (needs GPU for reasonable speed)
    """
    # Transparently handle URLs passed here too
    if str(file_path).startswith("http://") or str(file_path).startswith("https://"):
        return transcribe_url(str(file_path), model_size, language)

    path = Path(file_path)
    if not path.exists():
        logger.error(f"File not found: {path}")
        return None

    if not _whisper_supports(language):
        logger.warning(f"Whisper doesn't support '{language}' — skipping: {path}")
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


def transcribe_url(
    url: str,
    model_size: str = "base",
    language: str | None = None,
) -> TranscribedVideo | None:
    """
    Transcribe a video/audio from any URL — no manual downloading needed.
    ``language`` is the registry code the video is ingested under; Whisper is
    never run for a language it can't transcribe (e.g. Twi), whether that
    comes from ``language`` or from a jw.org URL.

    Supports:
      - JW.org hash-routed pages: https://www.jw.org/en/library/videos/#en/mediaitems/...
      - Direct links:  https://yoursite.com/video.mp4  (streamed, not saved)
      - YouTube:       https://youtube.com/watch?v=...
      - Vimeo:         https://vimeo.com/...
      - 1000+ sites via yt-dlp
    """
    logger.info(f"Processing video URL: {url}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # JW.org hash-routed pages: resolve via mediator API.
        jworg = _resolve_jworg_url(url)
        if jworg:
            # Fast path: VTT is a ready-made professional transcript — no Whisper needed.
            if jworg.get("vtt_url"):
                transcript = _fetch_vtt_transcript(jworg["vtt_url"])
                if transcript:
                    logger.success(f"[JW.org] Transcript from VTT: {url} ({len(transcript)} chars)")
                    return TranscribedVideo(
                        source=url,
                        title=jworg["title"] or _title_from_url(url),
                        transcript=transcript,
                        language=jworg["language"],
                        duration_seconds=jworg["duration"],
                        metadata={
                            "source_url": url,
                            "transcript_source": "vtt",
                            "file_type": "video",
                            "language": jworg["language"],
                        },
                    )

            # Whisper has no model for some languages (e.g. Twi) and would
            # produce a garbage transcript, so those videos need VTT subtitles.
            if not jworg["whisper"] or not _whisper_supports(language):
                logger.warning(
                    f"[JW.org] No VTT and Whisper doesn't support '{jworg['language']}' — skipping: {url}"
                )
                return None

            # Slow path: download smallest MP4 and run Whisper.
            if jworg.get("mp4_url"):
                logger.info(f"[JW.org] No VTT — downloading MP4 for Whisper: {jworg['mp4_url']}")
                audio_path = _download_direct(jworg["mp4_url"], tmp_dir)
                if audio_path:
                    result = transcribe_file(audio_path, model_size, language)
                    if result:
                        result.source = url
                        result.title = jworg["title"] or _title_from_url(url)
                        result.metadata["source_url"] = url
                    return result

            logger.error(f"JW.org mediator returned no usable URL for: {url}")
            return None

        # Non-JW.org URLs: direct file or yt-dlp. Check before downloading.
        if not _whisper_supports(language):
            logger.warning(f"Whisper doesn't support '{language}' — skipping: {url}")
            return None

        if _is_direct_media_url(url):
            audio_path = _download_direct(url, tmp_dir)
            if not audio_path:
                logger.info("Direct download failed, trying yt-dlp as fallback...")
                audio_path = _download_via_ytdlp(url, tmp_dir)
        else:
            audio_path = _download_via_ytdlp(url, tmp_dir)

        if not audio_path:
            logger.error(f"Could not download audio from: {url}")
            return None

        result = transcribe_file(audio_path, model_size, language)
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

def _whisper_supports(language: str | None) -> bool:
    """False for a registered language Whisper has no model for (e.g. Twi).
    No language, or an unregistered one, leaves it to Whisper's detection."""
    import languages

    lang = languages.LANGUAGES.get(language) if language else None
    return lang.whisper if lang else True


def _resolve_jworg_url(url: str) -> dict | None:
    """
    Resolve a JW.org hash-routed video page via the public mediator API.

    URL format:  https://www.jw.org/en/library/videos/#en/mediaitems/{cat}/{pub_id}
    API format:  https://b.jw-cdn.org/apis/mediator/v1/media-items/E/{pub_id}?clientType=www

    Returns {
        "title":    video title,
        "duration": seconds (float),
        "vtt_url":  subtitle/transcript URL (preferred — already plain text),
        "mp4_url":  lowest-quality MP4 URL (fallback for Whisper),
        "language": registry language code (or the raw locale if unregistered),
        "whisper":  whether Whisper can transcribe this language,
    } or None.
    """
    import re
    from urllib.parse import urlparse
    import languages

    parsed = urlparse(url)
    if "jw.org" not in (parsed.netloc or ""):
        return None

    # Fragment: e.g. "en/mediaitems/LatestVideos/pub-mwbv_202605_1_VIDEO"
    match = re.match(r"([a-z\-]+)/mediaitems/[^/]+/(.+)", parsed.fragment or "")
    if not match:
        return None

    lang_slug, pub_id = match.group(1), match.group(2)
    # Mediator API uses jw.org's own language codes ("en" → "E", "tw" → "TW",
    # "es" → "S"), which are not derivable from the locale, so registered
    # languages take theirs from the registry. Unregistered locales fall back
    # to the first letter, which is right for some (en, fr, de) but not all.
    lang = languages.from_jw_locale(lang_slug)
    api_lang = lang.jw_code if lang else lang_slug[:1].upper()
    language_code = lang.code if lang else lang_slug
    whisper_ok = lang.whisper if lang else True

    api_url = (
        f"https://b.jw-cdn.org/apis/mediator/v1/media-items"
        f"/{api_lang}/{pub_id}?clientType=www"
    )
    logger.debug(f"JW.org mediator API: {api_url}")

    try:
        import httpx

        resp = httpx.get(api_url, timeout=15, follow_redirects=True,
                         headers={"User-Agent": "RAG-Bot/1.0"})
        resp.raise_for_status()
        data = resp.json()

        media_item = (data.get("media") or [{}])[0]
        title    = media_item.get("title", "")
        duration = float(media_item.get("duration", 0.0))
        files    = media_item.get("files", [])

        # VTT subtitles are already a professional transcript — prefer over Whisper.
        # All quality variants share the same VTT, so grab from first file.
        vtt_url = (files[0].get("subtitles") or {}).get("url") if files else None

        # Smallest MP4 as Whisper fallback (only audio matters, not resolution).
        mp4s = sorted(
            [f for f in files if f.get("progressiveDownloadURL", "").endswith(".mp4")],
            key=lambda f: f.get("filesize", float("inf")),
        )
        mp4_url = mp4s[0]["progressiveDownloadURL"] if mp4s else None

        if vtt_url or mp4_url:
            return {
                "title": title,
                "duration": duration,
                "vtt_url": vtt_url,
                "mp4_url": mp4_url,
                "language": language_code,
                "whisper": whisper_ok,
            }

    except Exception as e:
        logger.warning(f"JW.org mediator API failed for {pub_id}: {e}")

    return None


def _fetch_vtt_transcript(vtt_url: str) -> str | None:
    """Download a WebVTT file and return its text content as a plain string."""
    try:
        import httpx
        resp = httpx.get(vtt_url, timeout=15, follow_redirects=True,
                         headers={"User-Agent": "RAG-Bot/1.0"})
        resp.raise_for_status()
        return _parse_vtt(resp.text)
    except Exception as e:
        logger.warning(f"VTT download failed ({vtt_url}): {e}")
        return None


def _parse_vtt(content: str) -> str:
    """Extract plain dialogue text from a WebVTT subtitle file."""
    import re

    lines = []
    for line in content.splitlines():
        line = line.strip()
        if (
            not line
            or line.startswith("WEBVTT")
            or line.startswith("NOTE")
            or re.match(r"^\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*", line)
            or re.match(r"^\d+$", line)          # bare cue number
        ):
            continue
        line = re.sub(r"<[^>]+>", "", line)      # strip inline VTT tags
        if line:
            lines.append(line)

    return " ".join(lines)


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
