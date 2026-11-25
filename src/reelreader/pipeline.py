"""The index pipeline: transcode, transcribe, segment, extract artifacts.

Every stage checks for its output first, so an interrupted run resumes at the
first missing piece instead of starting over.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from reelreader.library import Library, SegmentRecord, Video
from reelreader.segment import (
    Segment,
    SegmentError,
    build_segments,
    detect_scene_cuts,
    load_segments,
    save_segments,
    write_artifacts,
)
from reelreader.transcribe import (
    FasterWhisperTranscriber,
    Transcriber,
    TranscriptionError,
    empty_transcript,
    load_transcript,
    save_transcript,
)

PROXY_HEIGHT = 480


class PipelineError(RuntimeError):
    """A stage failed; the message names the stage and the fix."""


@dataclass(frozen=True, slots=True)
class IndexResult:
    video: Video
    ran: tuple[str, ...]
    skipped: tuple[str, ...]
    segment_count: int


def index_video(
    library: Library,
    video: Video,
    *,
    model: str = "small",
    device: str = "auto",
    compute_type: str = "int8",
    force: bool = False,
    transcriber: Transcriber | None = None,
    scene_cuts: list[float] | None = None,
    ffmpeg: str | None = None,
    log: Callable[[str], None] | None = None,
) -> IndexResult:
    emit = log or (lambda _message: None)
    out_dir = library.root / "videos" / video.id
    job = library.start_job(video.id, "index")
    library.set_status(video.id, "indexing")
    ran: list[str] = []
    skipped: list[str] = []
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        if not video.path.is_file():
            raise PipelineError(f"source file is missing: {video.path}")

        proxy = out_dir / "proxy.mp4"
        audio = out_dir / "audio.wav"
        needs_proxy = force or not proxy.exists()
        needs_audio = video.has_audio and (force or not audio.exists())
        if needs_proxy or needs_audio:
            emit(f"transcoding {video.title}")
            if needs_proxy:
                _ffmpeg(
                    [
                        "-i",
                        str(video.path),
                        "-vf",
                        f"scale=-2:{PROXY_HEIGHT}",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "28",
                        "-c:a",
                        "aac",
                        "-movflags",
                        "+faststart",
                        str(proxy),
                    ],
                    ffmpeg,
                    proxy,
                )
            if needs_audio:
                _ffmpeg(
                    [
                        "-i",
                        str(video.path),
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-c:a",
                        "pcm_s16le",
                        str(audio),
                    ],
                    ffmpeg,
                    audio,
                )
            ran.append("transcode")
        else:
            skipped.append("transcode")

        transcript_path = out_dir / "transcript.json"
        if transcript_path.exists() and not force:
            transcript = load_transcript(transcript_path)
            skipped.append("transcribe")
        elif not video.has_audio:
            emit("no audio track; skipping transcription")
            transcript = empty_transcript(video.duration)
            save_transcript(transcript_path, transcript)
            ran.append("transcribe")
        else:
            emit(f"transcribing {video.title} ({model})")
            engine = transcriber or FasterWhisperTranscriber(
                model,
                device=device,
                compute_type=compute_type,
            )
            transcript = engine.transcribe(audio, duration=video.duration)
            save_transcript(transcript_path, transcript)
            ran.append("transcribe")

        cuts = scene_cuts if scene_cuts is not None else detect_scene_cuts(proxy)

        segments_path = out_dir / "segments.json"
        if segments_path.exists() and not force:
            segments = load_segments(segments_path)
            skipped.append("segment")
        else:
            emit(f"segmenting {video.title}")
            segments = build_segments(
                transcript.sentences,
                cuts=cuts,
                duration=video.duration,
            )
            ran.append("segment")

        if not segments:
            raise PipelineError("no segments were produced")

        if force or not _artifacts_complete(out_dir, segments):
            emit(f"extracting {len(segments)} segment artifact(s)")
            segments = write_artifacts(
                segments,
                out_dir=out_dir,
                proxy=proxy,
                ffmpeg=ffmpeg,
                force=force,
                with_audio=video.has_audio,
            )
            ran.append("artifacts")
        else:
            skipped.append("artifacts")

        save_segments(
            segments_path,
            video_id=video.id,
            source_sha256=video.sha256,
            segments=segments,
        )
        library.replace_segments(
            video.id,
            [
                SegmentRecord(
                    id=segment.id,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    keyframe_at=segment.keyframe_at,
                )
                for segment in segments
            ],
        )
        library.set_status(video.id, "ready")
        library.finish_job(job.id, message=f"{len(segments)} segments")
    except KeyboardInterrupt:
        library.update_job(job.id, state="cancelled", message="interrupted")
        library.set_status(video.id, "interrupted")
        raise
    except Exception as exc:
        message = _message_for(exc)
        library.fail_job(job.id, message)
        library.set_status(video.id, "failed")
        if isinstance(exc, PipelineError):
            raise
        raise PipelineError(message) from exc

    return IndexResult(
        video=video,
        ran=tuple(ran),
        skipped=tuple(skipped),
        segment_count=len(segments),
    )


def _artifacts_complete(out_dir: Path, segments: list[Segment]) -> bool:
    if not segments:
        return False
    if not all(segment.artifacts for segment in segments):
        return False
    return all(
        (out_dir / relative).is_file()
        for segment in segments
        for relative in segment.artifacts.values()
    )


def _message_for(exc: Exception) -> str:
    if isinstance(exc, (PipelineError, SegmentError, TranscriptionError, OSError)):
        return str(exc)
    return f"indexing failed: {exc}"


def _ffmpeg(args: list[str], binary: str | None, destination: Path) -> None:
    executable = binary or shutil.which("ffmpeg")
    if executable is None:
        raise PipelineError("ffmpeg not found. Install it, for example: brew install ffmpeg")
    command = [executable, "-y", "-v", "error", *args]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    except FileNotFoundError as exc:
        raise PipelineError(f"could not run {executable}") from exc
    except subprocess.TimeoutExpired as exc:
        raise PipelineError(f"ffmpeg timed out writing {destination.name}") from exc
    if completed.returncode != 0:
        detail = " ".join((completed.stderr or "").split())[:300]
        raise PipelineError(f"ffmpeg failed writing {destination.name}: {detail or 'no output'}")
    if not destination.exists() or destination.stat().st_size == 0:
        raise PipelineError(f"ffmpeg produced no {destination.name}")
