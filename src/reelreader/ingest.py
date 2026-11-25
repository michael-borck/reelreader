"""Bring a video into the library.

The original file is referenced in place by default. ``copy=True`` brings a
copy into ``videos/<id>/source.*`` so the library is self-contained. Either
way, deduplication is by SHA-256, so the same lecture added twice is a no-op.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from reelreader.ids import new_id
from reelreader.library import Library, Video
from reelreader.media import MediaInfo, probe

VIDEO_EXTENSIONS = {
    ".mp4",
    ".m4v",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".flv",
}

Probe = Callable[[Path], MediaInfo]


class IngestError(RuntimeError):
    """The source cannot be ingested, with a reason the user can act on."""


@dataclass(frozen=True, slots=True)
class IngestResult:
    video: Video
    created: bool
    skipped: str | None = None


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_url(source: str) -> bool:
    return urlparse(source).scheme in {"http", "https"}


def ingest_path(
    library: Library,
    path: Path | str,
    *,
    title: str | None = None,
    copy: bool = False,
    probe_fn: Probe | None = None,
) -> list[IngestResult]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise IngestError(f"{source} does not exist")
    if source.is_file():
        return [ingest_file(library, source, title=title, copy=copy, probe_fn=probe_fn)]
    if not source.is_dir():
        raise IngestError(f"{source} is neither a file nor a folder")
    files = sorted(
        item
        for item in source.rglob("*")
        if item.is_file()
        and item.suffix.lower() in VIDEO_EXTENSIONS
        and not item.name.startswith(".")
    )
    if not files:
        raise IngestError(f"no video files found in {source}")
    return [ingest_file(library, item, copy=copy, probe_fn=probe_fn) for item in files]


def ingest_file(
    library: Library,
    path: Path | str,
    *,
    title: str | None = None,
    copy: bool = False,
    source: str = "file",
    video_id: str | None = None,
    probe_fn: Probe | None = None,
) -> IngestResult:
    original = Path(path).expanduser().resolve()
    if not original.is_file():
        raise IngestError(f"{original} does not exist")
    suffix = original.suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise IngestError(f"{suffix or 'no extension'} is not a video file ({supported})")

    digest = sha256_file(original)
    existing = library.find_by_sha(digest)
    if existing is not None:
        return IngestResult(video=existing, created=False, skipped="already in library")

    inspector = probe_fn or probe
    try:
        info = inspector(original)
    except Exception as exc:
        raise IngestError(f"could not read {original.name}: {exc}") from exc

    identifier = video_id or new_id()
    stored_path = original
    copied = False
    if copy:
        destination_dir = library.root / "videos" / identifier
        destination_dir.mkdir(parents=True, exist_ok=True)
        stored_path = destination_dir / f"source{suffix}"
        shutil.copy2(original, stored_path)
        copied = True

    video, created = library.add_video(
        path=stored_path,
        sha256=digest,
        title=title or original.stem,
        duration=info.duration,
        width=info.width,
        height=info.height,
        has_audio=info.has_audio,
        source=source,
        video_id=identifier,
    )
    if not created and copied:
        stored_path.unlink(missing_ok=True)
    return IngestResult(video=video, created=created)


def ingest_url(
    library: Library,
    url: str,
    *,
    title: str | None = None,
    probe_fn: Probe | None = None,
) -> IngestResult:
    binary = shutil.which("yt-dlp")
    if binary is None:
        raise IngestError("URL ingest needs yt-dlp. Install it: brew install yt-dlp")
    identifier = new_id()
    destination = library.root / "videos" / identifier
    destination.mkdir(parents=True, exist_ok=True)
    template = str(destination / "source.%(ext)s")
    command = [
        binary,
        "--no-playlist",
        "--no-progress",
        "-f",
        "bv*+ba/b",
        "-o",
        template,
        url,
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    except FileNotFoundError as exc:
        raise IngestError("could not run yt-dlp") from exc
    except subprocess.TimeoutExpired as exc:
        raise IngestError("yt-dlp timed out after an hour") from exc
    if completed.returncode != 0:
        detail = " ".join((completed.stderr or "").split())[-300:]
        raise IngestError(f"yt-dlp failed: {detail or 'no output'}")
    candidates = [
        item
        for item in destination.glob("source.*")
        if item.suffix.lower() in VIDEO_EXTENSIONS and item.stat().st_size > 0
    ]
    if not candidates:
        raise IngestError("yt-dlp reported success but wrote no video file")
    downloaded = max(candidates, key=lambda item: item.stat().st_size)
    return ingest_file(
        library,
        downloaded,
        title=title,
        source="url",
        video_id=identifier,
        probe_fn=probe_fn,
    )
