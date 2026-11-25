"""Turn sentences into 20–40 second moments with playable artifacts.

Segments end on a sentence, a scene change, or a long silence. Each one gets a
clip, a wav, a keyframe, and a JSON record so later stages never recompute
boundaries.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path

from reelreader.ids import new_id
from reelreader.transcribe import Sentence

MIN_SECONDS = 18.0
MAX_SECONDS = 42.0
BOUNDARY_GAP_SECONDS = 1.2
MAX_GAP_SECONDS = 4.0


class SegmentError(RuntimeError):
    """Segmentation or artifact extraction failed."""


@dataclass(frozen=True, slots=True)
class Segment:
    index: int
    id: str
    start: float
    end: float
    text: str
    keyframe_at: float
    artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "keyframe_at": self.keyframe_at,
            "artifacts": dict(self.artifacts),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Segment:
        return cls(
            index=int(data["index"]),
            id=str(data["id"]),
            start=float(data["start"]),
            end=float(data["end"]),
            text=str(data.get("text", "")),
            keyframe_at=float(data.get("keyframe_at", data["start"])),
            artifacts=dict(data.get("artifacts") or {}),
        )


def build_segments(
    sentences: list[Sentence] | tuple[Sentence, ...],
    *,
    cuts: list[float] | tuple[float, ...] = (),
    duration: float = 0.0,
    min_seconds: float = MIN_SECONDS,
    max_seconds: float = MAX_SECONDS,
    gap_seconds: float = BOUNDARY_GAP_SECONDS,
) -> list[Segment]:
    usable = [
        Sentence(
            start=max(0.0, sentence.start),
            end=max(sentence.start, sentence.end if duration <= 0 else min(sentence.end, duration)),
            text=sentence.text.strip(),
        )
        for sentence in sentences
        if sentence.text.strip()
    ]
    usable = [sentence for sentence in usable if sentence.end > sentence.start]
    if not usable:
        if duration <= 0:
            return []
        return [
            Segment(
                index=1,
                id=new_id(),
                start=0.0,
                end=duration,
                text="",
                keyframe_at=duration / 2,
            )
        ]

    groups: list[list[Sentence]] = [[usable[0]]]
    for nxt in usable[1:]:
        current = groups[-1]
        start = current[0].start
        end = current[-1].end
        span = end - start
        gap = nxt.start - end
        at_cut = any(end - 0.25 <= cut <= nxt.start + 0.25 for cut in cuts)
        projected = nxt.end - start
        should_close = (
            span >= min_seconds and (projected > max_seconds or at_cut or gap >= gap_seconds)
        ) or gap > MAX_GAP_SECONDS
        if should_close:
            groups.append([nxt])
        else:
            current.append(nxt)

    if len(groups) > 1 and _span(groups[-1]) < min_seconds * 0.6:
        tail = groups.pop()
        groups[-1].extend(tail)

    return [
        Segment(
            index=index,
            id=new_id(),
            start=group[0].start,
            end=group[-1].end,
            text=" ".join(item.text for item in group).strip(),
            keyframe_at=(group[0].start + group[-1].end) / 2,
        )
        for index, group in enumerate(groups, start=1)
    ]


def detect_scene_cuts(path: Path) -> list[float]:
    import importlib

    try:
        module = importlib.import_module("scenedetect")
    except ImportError:
        return []
    try:
        scenes = module.detect(str(path), module.ContentDetector())
    except Exception:
        return []
    cuts: list[float] = []
    for start, _ in scenes:
        seconds = float(start.get_seconds())
        if seconds > 0.1:
            cuts.append(seconds)
    return cuts


def write_artifacts(
    segments: list[Segment],
    *,
    out_dir: Path,
    proxy: Path,
    ffmpeg: str | None = None,
    force: bool = False,
    with_audio: bool = True,
    run=subprocess.run,
) -> list[Segment]:
    binary = ffmpeg or shutil.which("ffmpeg")
    if binary is None:
        raise SegmentError("ffmpeg not found. Install it, for example: brew install ffmpeg")
    clips_dir = out_dir / "segments"
    clips_dir.mkdir(parents=True, exist_ok=True)
    updated: list[Segment] = []
    for segment in segments:
        stem = f"{segment.index:04d}"
        clip = clips_dir / f"{stem}.mp4"
        audio = clips_dir / f"{stem}.wav"
        frame = clips_dir / f"{stem}.jpg"
        if force:
            for path in (clip, audio, frame):
                path.unlink(missing_ok=True)
        duration = max(0.1, segment.duration)
        if not clip.exists():
            _ffmpeg(
                [
                    "-ss",
                    f"{segment.start:.3f}",
                    "-i",
                    str(proxy),
                    "-t",
                    f"{duration:.3f}",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-crf",
                    "30",
                    "-an",
                    "-movflags",
                    "+faststart",
                    str(clip),
                ],
                binary,
                run,
                clip,
            )
        if with_audio and not audio.exists():
            _ffmpeg(
                [
                    "-ss",
                    f"{segment.start:.3f}",
                    "-i",
                    str(proxy),
                    "-t",
                    f"{duration:.3f}",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(audio),
                ],
                binary,
                run,
                audio,
            )
        if not frame.exists():
            _ffmpeg(
                [
                    "-ss",
                    f"{segment.keyframe_at:.3f}",
                    "-i",
                    str(proxy),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "3",
                    str(frame),
                ],
                binary,
                run,
                frame,
            )
        artifacts = {
            "clip": str(clip.relative_to(out_dir)),
            "keyframe": str(frame.relative_to(out_dir)),
        }
        if with_audio:
            artifacts["audio"] = str(audio.relative_to(out_dir))
        updated.append(replace(segment, artifacts=artifacts))
    return updated


def save_segments(
    path: Path, *, video_id: str, source_sha256: str, segments: list[Segment]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": 1,
        "video_id": video_id,
        "source_sha256": source_sha256,
        "segments": [segment.to_dict() for segment in segments],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_segments(path: Path) -> list[Segment]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [Segment.from_dict(item) for item in data["segments"]]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SegmentError(f"cannot read {path}: {exc}") from exc


def _ffmpeg(args: list[str], binary: str, run, destination: Path) -> None:
    command = [binary, "-y", "-v", "error", *args]
    try:
        completed = run(command, capture_output=True, text=True, timeout=600)
    except FileNotFoundError as exc:
        raise SegmentError(f"could not run {binary}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SegmentError(f"ffmpeg timed out writing {destination.name}") from exc
    if completed.returncode != 0:
        detail = " ".join((completed.stderr or "").split())[:300]
        raise SegmentError(f"ffmpeg failed writing {destination.name}: {detail or 'no output'}")
    if not destination.exists() or destination.stat().st_size == 0:
        raise SegmentError(f"ffmpeg produced no {destination.name}")


def _span(group: list[Sentence]) -> float:
    return group[-1].end - group[0].start
