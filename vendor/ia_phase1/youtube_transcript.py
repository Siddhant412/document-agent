from __future__ import annotations

import html
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import yt_dlp
from yt_dlp.utils import DownloadError

logger = logging.getLogger(__name__)

_SUB_EXTS = {".vtt", ".srt"}
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SRT_INDEX_RE = re.compile(r"^\d+$")


def _default_transcript_dir() -> Path:
    configured = os.getenv("IA_PHASE1_TRANSCRIPT_OUTPUT_DIR", "").strip()
    if not configured:
        configured = os.getenv("YOUTUBE_TRANSCRIPT_OUTPUT_DIR", "").strip()
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        root = (Path.cwd() / ".ia_phase1_data" / "transcripts").expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def extract_youtube_video_id(source: str) -> Optional[str]:
    raw = (source or "").strip()
    if not raw:
        return None
    try:
        parsed = urlparse(raw)
    except Exception:
        return None
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]

    if host == "youtu.be":
        parts = [p for p in (parsed.path or "").split("/") if p]
        return parts[0] if parts else None

    if host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        path = (parsed.path or "").strip("/")
        if path == "watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        parts = [p for p in path.split("/") if p]
        if not parts:
            return None
        if parts[0] in {"shorts", "live", "embed"} and len(parts) >= 2:
            return parts[1]
    return None


def is_youtube_url(source: str) -> bool:
    return bool(extract_youtube_video_id(source))


def _clear_stale_caption_files(video_dir: Path, video_id: str) -> None:
    for stale in video_dir.glob(f"{video_id}*"):
        if stale.is_file() and stale.suffix.lower() in {".vtt", ".srt", ".txt"}:
            try:
                stale.unlink(missing_ok=True)
            except Exception:
                logger.debug("Failed to remove stale transcript artifact %s", stale, exc_info=True)


def _collect_caption_files(video_dir: Path, video_id: str) -> List[Path]:
    matches: List[Path] = []
    for item in video_dir.glob(f"{video_id}*"):
        if item.is_file() and item.suffix.lower() in _SUB_EXTS:
            matches.append(item)
    return sorted(matches)


def _subtitle_score(path: Path) -> int:
    name = path.name.lower()
    score = 0
    if path.suffix.lower() == ".vtt":
        score += 20
    if ".en" in name:
        score += 35
    if ".en-us" in name or ".en-gb" in name:
        score += 8
    if "auto" in name or "asr" in name:
        score -= 6
    return score


def _pick_caption_file(files: List[Path]) -> Optional[Path]:
    if not files:
        return None
    return sorted(files, key=lambda p: (_subtitle_score(p), -len(p.name)), reverse=True)[0]


def _clean_caption_text(raw: str) -> str:
    cleaned: List[str] = []
    in_note = False
    for source_line in (raw or "").splitlines():
        line = source_line.replace("\ufeff", "").strip()
        if not line:
            in_note = False
            continue

        upper = line.upper()
        if upper.startswith("WEBVTT") or upper.startswith("KIND:") or upper.startswith("LANGUAGE:"):
            continue
        if upper.startswith("NOTE"):
            in_note = True
            continue
        if in_note:
            continue
        if "-->" in line:
            continue
        if _SRT_INDEX_RE.match(line):
            continue

        line = _HTML_TAG_RE.sub("", line)
        line = html.unescape(line)
        line = " ".join(line.split()).strip()
        if not line:
            continue

        if cleaned:
            prev = cleaned[-1]
            if line == prev:
                continue
            # Collapse rolling caption updates: keep the fuller variant.
            if line.startswith(prev):
                cleaned[-1] = line
                continue
            if prev.startswith(line):
                continue

        cleaned.append(line)
    return "\n".join(cleaned).strip()


def download_youtube_transcript(
    video_url: str,
    output_dir: Optional[Path] = None,
    preferred_langs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Download YouTube subtitles (manual/auto) via yt-dlp and persist a cleaned transcript text file.
    """
    root = (output_dir or _default_transcript_dir()).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    langs = preferred_langs or ["en", "en-US", "en-GB"]
    attempt_lang_sets: List[List[str]] = [langs, ["all"]]

    info: Dict[str, Any] = {}
    video_id: Optional[str] = None
    selected_caption: Optional[Path] = None
    video_dir: Optional[Path] = None

    for lang_set in attempt_lang_sets:
        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": lang_set,
            "subtitlesformat": "vtt/srt/best",
            "outtmpl": str(root / "%(id)s" / "%(id)s.%(ext)s"),
            "restrictfilenames": True,
            "overwrites": True,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "ignoreconfig": True,
        }

        download_error: Optional[Exception] = None
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            extracted = ydl.extract_info(video_url, download=False) or {}
            if not isinstance(extracted, dict):
                raise RuntimeError("Unable to read YouTube metadata.")
            info = extracted
            video_id = str(info.get("id") or "").strip() or extract_youtube_video_id(video_url)
            if not video_id:
                raise RuntimeError("Could not determine YouTube video id.")

            video_dir = root / video_id
            video_dir.mkdir(parents=True, exist_ok=True)
            _clear_stale_caption_files(video_dir, video_id)
            try:
                ydl.download([video_url])
            except DownloadError as exc:
                # Some per-language downloads can fail (rate-limit/region), while other caption files
                # are still saved successfully. We only fail if no usable caption file exists.
                download_error = exc

        caption_files = _collect_caption_files(video_dir, video_id)
        selected_caption = _pick_caption_file(caption_files)
        if selected_caption:
            break
        if download_error is not None:
            logger.warning(
                "Subtitle download attempt failed for %s with langs=%s: %s",
                video_url,
                lang_set,
                download_error,
            )

    if not video_id or not video_dir:
        raise RuntimeError("Could not initialize transcript workspace.")
    if not selected_caption or not selected_caption.exists():
        raise RuntimeError("No subtitles available for this YouTube video.")

    caption_raw = selected_caption.read_text(encoding="utf-8", errors="ignore")
    transcript_body = _clean_caption_text(caption_raw)
    if not transcript_body:
        raise RuntimeError("Subtitle extraction produced empty transcript text.")

    title = str(info.get("title") or f"YouTube Video {video_id}").strip() or f"YouTube Video {video_id}"
    transcript_path = video_dir / f"{video_id}.txt"
    transcript_text = (
        f"Title: {title}\n"
        f"URL: {video_url}\n"
        f"Video ID: {video_id}\n"
        f"Caption File: {selected_caption.name}\n\n"
        f"{transcript_body}\n"
    )
    transcript_path.write_text(transcript_text, encoding="utf-8")

    return {
        "video_id": video_id,
        "title": title,
        "video_url": video_url,
        "subtitle_path": str(selected_caption),
        "transcript_path": str(transcript_path),
        "transcript_text": transcript_text,
    }


__all__ = [
    "extract_youtube_video_id",
    "is_youtube_url",
    "download_youtube_transcript",
]
