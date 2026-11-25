"""Media inspection through ffprobe."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class ProbeError(RuntimeError):
    """ffprobe is missing, failed, or the file has no usable video stream."""


@dataclass(frozen=True, slots=True)
class MediaInfo:
    duration: float
    width: int | None
    height: int | None
    fps: float | None
    has_audio: bool
    video_codec: str | None
    audio_codec: str | None


def probe(
    path: Path | str,
    *,
    ffprobe: str | None = None,
    run=subprocess.run,
) -> MediaInfo:
    source = Path(path)
    if not source.is_file():
        raise ProbeError(f"{source} does not exist")
    binary = ffprobe or shutil.which("ffprobe")
    if binary is None:
        raise ProbeError("ffprobe not found. Install ffmpeg, for example: brew install ffmpeg")
    command = [
        binary,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(source),
    ]
    try:
        completed = run(command, capture_output=True, text=True, timeout=60)
    except FileNotFoundError as exc:
        raise ProbeError(f"could not run {binary}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ProbeError(f"ffprobe timed out on {source}") from exc
    if completed.returncode != 0:
        detail = " ".join((completed.stderr or "").split())[:300]
        raise ProbeError(f"ffprobe failed on {source}: {detail or 'no output'}")
    try:
        data = json.loads(completed.stdout or "{}")
    except ValueError as exc:
        raise ProbeError(f"ffprobe returned invalid JSON for {source}") from exc
    return parse_probe(data, source)


def parse_probe(data: dict, source: Path) -> MediaInfo:
    streams = data.get("streams") or []
    video = next(
        (
            stream
            for stream in streams
            if stream.get("codec_type") == "video"
            and not (stream.get("disposition") or {}).get("attached_pic")
        ),
        None,
    )
    if video is None:
        raise ProbeError(f"no video stream in {source}")
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    container = data.get("format") or {}
    duration = _number(container.get("duration")) or _number(video.get("duration")) or 0.0
    return MediaInfo(
        duration=duration,
        width=_integer(video.get("width")),
        height=_integer(video.get("height")),
        fps=_ratio(video.get("avg_frame_rate")) or _ratio(video.get("r_frame_rate")),
        has_audio=audio is not None,
        video_codec=video.get("codec_name"),
        audio_codec=audio.get("codec_name") if audio else None,
    )


def _number(value: object) -> float | None:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _integer(value: object) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _ratio(value: object) -> float | None:
    if not isinstance(value, str) or "/" not in value:
        return None
    numerator, _, denominator = value.partition("/")
    try:
        top = float(numerator)
        bottom = float(denominator)
    except ValueError:
        return None
    if bottom == 0:
        return None
    return round(top / bottom, 3)
