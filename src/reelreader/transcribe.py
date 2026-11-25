"""Speech to text with word-level timestamps.

Word times are the unit of citation precision; sentences exist so segments can
end on something a person would recognise. The transcript is written to
``transcript.json`` and is the source the segmenter reads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

SENTENCE_ENDINGS = ".?!…。！？"
MAX_SENTENCE_CHARS = 240
MAX_SENTENCE_SECONDS = 25.0
SENTENCE_GAP_SECONDS = 0.8


class TranscriptionError(RuntimeError):
    """Speech recognition is unavailable or failed, with a fix the user can act on."""


@dataclass(frozen=True, slots=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class Sentence:
    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class Transcript:
    language: str | None
    engine: str
    duration: float
    words: tuple[Word, ...]
    sentences: tuple[Sentence, ...]

    def to_dict(self) -> dict:
        return {
            "format": 1,
            "language": self.language,
            "engine": self.engine,
            "duration": self.duration,
            "words": [{"start": w.start, "end": w.end, "text": w.text} for w in self.words],
            "sentences": [{"start": s.start, "end": s.end, "text": s.text} for s in self.sentences],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Transcript:
        try:
            words = tuple(
                Word(float(item["start"]), float(item["end"]), str(item["text"]))
                for item in data.get("words", [])
            )
            sentences = tuple(
                Sentence(float(item["start"]), float(item["end"]), str(item["text"]))
                for item in data.get("sentences", [])
            )
            return cls(
                language=data.get("language"),
                engine=str(data.get("engine") or "unknown"),
                duration=float(data.get("duration") or 0.0),
                words=words,
                sentences=sentences,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TranscriptionError(f"invalid transcript.json: {exc}") from exc


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio: Path, *, duration: float | None = None) -> Transcript: ...


def build_sentences(
    words: list[Word] | tuple[Word, ...],
    *,
    max_chars: int = MAX_SENTENCE_CHARS,
    max_seconds: float = MAX_SENTENCE_SECONDS,
    gap_seconds: float = SENTENCE_GAP_SECONDS,
) -> list[Sentence]:
    sentences: list[Sentence] = []
    buffer: list[Word] = []

    def flush() -> None:
        if not buffer:
            return
        text = " ".join(word.text for word in buffer if word.text).strip()
        if text:
            sentences.append(Sentence(start=buffer[0].start, end=buffer[-1].end, text=text))
        buffer.clear()

    for word in words:
        if not word.text.strip():
            continue
        if buffer:
            current_text = " ".join(item.text for item in buffer)
            too_long = (
                len(current_text) >= max_chars or buffer[-1].end - buffer[0].start >= max_seconds
            )
            if too_long or word.start - buffer[-1].end >= gap_seconds:
                flush()
        buffer.append(word)
        if word.text.rstrip()[-1:] in SENTENCE_ENDINGS:
            flush()
    flush()
    return sentences


def load_transcript(path: Path) -> Transcript:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TranscriptionError(f"{path} does not exist") from exc
    except ValueError as exc:
        raise TranscriptionError(f"{path} is not valid JSON: {exc}") from exc
    return Transcript.from_dict(data)


def save_transcript(path: Path, transcript: Transcript) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(transcript.to_dict(), indent=2), encoding="utf-8")


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_size: str = "small",
        *,
        device: str = "auto",
        compute_type: str = "int8",
        download_root: Path | None = None,
    ) -> None:
        self.model_size = model_size
        self.name = f"faster-whisper:{model_size}"
        self._device = device
        self._compute_type = compute_type
        self._download_root = str(download_root) if download_root else None
        self._model = None

    def _load(self):
        import importlib

        try:
            module = importlib.import_module("faster_whisper")
        except ImportError as exc:
            raise TranscriptionError(
                "speech recognition needs the optional dependency: pip install 'reelreader[asr]'"
            ) from exc
        return module.WhisperModel(
            self.model_size,
            device=self._device,
            compute_type=self._compute_type,
            download_root=self._download_root,
        )

    def transcribe(self, audio: Path, *, duration: float | None = None) -> Transcript:
        model = self._model or self._load()
        self._model = model
        try:
            segments, info = model.transcribe(
                str(audio),
                word_timestamps=True,
                vad_filter=True,
            )
            words: list[Word] = []
            for segment in segments:
                for word in segment.words or []:
                    text = word.word.strip()
                    if text:
                        words.append(Word(start=float(word.start), end=float(word.end), text=text))
        except Exception as exc:
            raise TranscriptionError(f"{self.name} failed: {exc}") from exc
        return Transcript(
            language=getattr(info, "language", None),
            engine=self.name,
            duration=float(duration if duration is not None else getattr(info, "duration", 0.0)),
            words=tuple(words),
            sentences=tuple(build_sentences(words)),
        )


def empty_transcript(duration: float) -> Transcript:
    return Transcript(
        language=None,
        engine="none",
        duration=duration,
        words=(),
        sentences=(),
    )
