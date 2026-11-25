"""Text embeddings and the library index manifest.

The index is derived data: one matrix per video, plus a manifest that records
the model, dimension, and the segment ids in row order. A model change is an
error, never a silent mix.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from reelreader.library import Library, Video
from reelreader.providers.embedders import Embedder

INDEX_FORMAT = 1
DEFAULT_VIEW = "text"
MANIFEST_NAME = "manifest.json"


class EmbeddingError(RuntimeError):
    """Embedding or index writing failed, with a fix the user can act on."""


class IndexUnavailableError(EmbeddingError):
    """No index exists yet for this library."""


class IndexStaleError(EmbeddingError):
    """The index was built with a different model or dimension."""


@dataclass
class IndexManifest:
    model: str
    dim: int
    normalised: bool = True
    view: str = DEFAULT_VIEW
    format: int = INDEX_FORMAT
    created_at: str = ""
    videos: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "view": self.view,
            "model": self.model,
            "dim": self.dim,
            "normalised": self.normalised,
            "created_at": self.created_at,
            "videos": self.videos,
        }

    @classmethod
    def from_dict(cls, data: dict) -> IndexManifest:
        if int(data.get("format", 0)) != INDEX_FORMAT:
            raise EmbeddingError(
                f"unsupported index format {data.get('format')!r}; rebuild the index"
            )
        try:
            return cls(
                model=str(data["model"]),
                dim=int(data["dim"]),
                normalised=bool(data.get("normalised", True)),
                view=str(data.get("view", DEFAULT_VIEW)),
                format=INDEX_FORMAT,
                created_at=str(data.get("created_at", "")),
                videos=dict(data.get("videos") or {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError(f"invalid {MANIFEST_NAME}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class EmbedSummary:
    embedded: tuple[str, ...]
    skipped: tuple[str, ...]


def index_dir(library: Library) -> Path:
    return library.root / "index"


def manifest_path(library: Library) -> Path:
    return index_dir(library) / MANIFEST_NAME


def load_manifest(library: Library) -> IndexManifest | None:
    path = manifest_path(library)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise EmbeddingError(f"{path} is not valid JSON: {exc}") from exc
    return IndexManifest.from_dict(data)


def save_manifest(library: Library, manifest: IndexManifest) -> None:
    directory = index_dir(library)
    directory.mkdir(parents=True, exist_ok=True)
    manifest.created_at = manifest.created_at or _now()
    target = manifest_path(library)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    temporary.replace(target)


def ensure_compatible(manifest: IndexManifest, embedder: Embedder) -> None:
    if manifest.model != embedder.model or manifest.dim != embedder.dim:
        raise IndexStaleError(
            f"The index was built with {manifest.model} ({manifest.dim}d); "
            f"the configured embedder is {embedder.model} ({embedder.dim}d). "
            "Run: reelreader index --rebuild"
        )


def normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def embed_texts(embedder: Embedder, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, embedder.dim), dtype=np.float32)
    vectors = np.asarray(embedder.embed_texts(texts), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise EmbeddingError(f"{embedder.name} returned {vectors.shape} for {len(texts)} input(s)")
    if vectors.shape[1] != embedder.dim:
        raise EmbeddingError(
            f"{embedder.name} returned {vectors.shape[1]}-dimensional vectors but "
            f"embed.dim is {embedder.dim}. Set embed.dim to {vectors.shape[1]}."
        )
    return normalise(vectors)


def segment_input(title: str, text: str) -> str:
    cleaned = text.strip()
    return f"{title}\n{cleaned}" if cleaned else title


def embed_video(
    library: Library,
    video: Video,
    embedder: Embedder,
    *,
    manifest: IndexManifest | None = None,
    log=None,
) -> IndexManifest:
    emit = log or (lambda _message: None)
    segments = library.list_segments(video.id)
    if not segments:
        raise EmbeddingError(f"{video.title} has no segments; run reelreader index first")
    current = manifest or load_manifest(library)
    if current is None:
        current = IndexManifest(model=embedder.model, dim=embedder.dim)
    else:
        ensure_compatible(current, embedder)

    if current.dim != embedder.dim:
        raise EmbeddingError(
            f"manifest dim {current.dim} does not match embedder dim {embedder.dim}"
        )

    emit(f"embedding {video.title} ({len(segments)} segments)")
    vectors = embed_texts(
        embedder,
        [segment_input(video.title, segment.text) for segment in segments],
    )
    target = index_dir(library) / video.id / f"{DEFAULT_VIEW}.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, vectors)
    current.videos[video.id] = {
        "sha256": video.sha256,
        "count": len(segments),
        "segment_ids": [segment.id for segment in segments],
        "file": f"{video.id}/{DEFAULT_VIEW}.npy",
    }
    save_manifest(library, current)
    return current


def ensure_embeddings(
    library: Library,
    videos: list[Video],
    embedder: Embedder,
    *,
    force: bool = False,
    log=None,
) -> EmbedSummary:
    emit = log or (lambda _message: None)
    manifest = load_manifest(library)
    if manifest is not None and (manifest.model != embedder.model or manifest.dim != embedder.dim):
        if not force:
            ensure_compatible(manifest, embedder)
        emit("embedding model changed; rebuilding the whole index")
        clear_index(library)
        manifest = None
        videos = library.list_videos()

    embedded: list[str] = []
    skipped: list[str] = []
    for video in videos:
        segments = library.list_segments(video.id)
        if not segments:
            raise EmbeddingError(f"{video.title} has no segments; run reelreader index first")
        if not force and manifest is not None and _is_current(library, manifest, video, segments):
            skipped.append(video.id)
            continue
        job = library.start_job(video.id, "embed")
        try:
            manifest = embed_video(library, video, embedder, manifest=manifest, log=emit)
        except Exception as exc:
            library.fail_job(job.id, str(exc))
            raise
        library.finish_job(job.id, message=f"{len(segments)} segments")
        embedded.append(video.id)
    return EmbedSummary(tuple(embedded), tuple(skipped))


def clear_index(library: Library) -> None:
    directory = index_dir(library)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def load_vectors(library: Library, entry: dict) -> np.ndarray:
    relative = entry.get("file")
    if not relative:
        raise EmbeddingError("manifest entry has no file")
    path = index_dir(library) / relative
    try:
        return np.load(path).astype(np.float32)
    except OSError as exc:
        raise EmbeddingError(f"cannot read {path}: {exc}") from exc


def _is_current(library: Library, manifest: IndexManifest, video: Video, segments) -> bool:
    entry = manifest.videos.get(video.id)
    if not entry:
        return False
    if entry.get("sha256") != video.sha256:
        return False
    if entry.get("segment_ids") != [segment.id for segment in segments]:
        return False
    return (index_dir(library) / str(entry.get("file", ""))).is_file()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
