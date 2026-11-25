"""Retrieve moments by meaning, by keyword, or both.

Dense cosine finds paraphrases; BM25 finds names and numbers. Hybrid fuses the
two rankings with reciprocal rank fusion.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from reelreader.embed import (
    IndexManifest,
    IndexStaleError,
    IndexUnavailableError,
    ensure_compatible,
    load_manifest,
    load_vectors,
)
from reelreader.library import Library, SegmentRecord, Video
from reelreader.providers.embedders import Embedder

RRF_K = 60
MODES = ("hybrid", "dense", "bm25")
_TOKEN = re.compile(r"[a-z0-9]+")


class SearchError(RuntimeError):
    """The query or the index cannot be used, with a fix the user can act on."""


@dataclass(frozen=True, slots=True)
class Hit:
    segment_id: str
    video_id: str
    video_title: str
    start: float
    end: float
    text: str
    score: float
    dense: float | None = None
    bm25: float | None = None

    @property
    def label(self) -> str:
        return format_time(self.start)

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "video_id": self.video_id,
            "video_title": self.video_title,
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "end_label": format_time(self.end),
            "text": self.text,
            "score": round(self.score, 6),
            "dense": None if self.dense is None else round(self.dense, 6),
            "bm25": None if self.bm25 is None else round(self.bm25, 6),
        }


def format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def search_library(
    library: Library,
    query: str,
    *,
    embedder: Embedder,
    limit: int = 8,
    mode: str = "hybrid",
    video_id: str | None = None,
) -> list[Hit]:
    text = query.strip()
    if not text:
        return []
    if mode not in MODES:
        raise SearchError(f"mode must be one of {', '.join(MODES)}, got {mode!r}")
    if limit <= 0:
        raise SearchError("limit must be positive")

    manifest = load_manifest(library)
    if manifest is None:
        raise IndexUnavailableError("No index yet. Run: reelreader index")
    ensure_compatible(manifest, embedder)

    records, matrix = _load_corpus(library, manifest, video_id)
    if not records:
        return []

    dense = matrix @ _query_vector(embedder, text, manifest)
    bm25 = _bm25_scores([segment.text for _, segment in records], text)
    order, scores = _rank(dense, bm25, mode)

    hits: list[Hit] = []
    for index in order[:limit]:
        video, segment = records[index]
        hits.append(
            Hit(
                segment_id=segment.id,
                video_id=video.id,
                video_title=video.title,
                start=segment.start,
                end=segment.end,
                text=segment.text,
                score=float(scores[index]),
                dense=float(dense[index]),
                bm25=float(bm25[index]),
            )
        )
    return hits


def _load_corpus(
    library: Library,
    manifest: IndexManifest,
    video_id: str | None,
) -> tuple[list[tuple[Video, SegmentRecord]], np.ndarray]:
    records: list[tuple[Video, SegmentRecord]] = []
    rows: list[np.ndarray] = []
    for identifier, entry in manifest.videos.items():
        if video_id is not None and identifier != video_id:
            continue
        try:
            video = library.get_video(identifier)
        except Exception:
            continue
        segments = {segment.id: segment for segment in library.list_segments(identifier)}
        matrix = load_vectors(library, entry)
        for row, segment_id in enumerate(entry.get("segment_ids", [])):
            segment = segments.get(segment_id)
            if segment is None or row >= matrix.shape[0]:
                continue
            records.append((video, segment))
            rows.append(matrix[row])
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    return records, np.vstack(rows).astype(np.float32)


def _query_vector(embedder: Embedder, query: str, manifest: IndexManifest) -> np.ndarray:
    vector = np.asarray(embedder.embed_texts([query])[0], dtype=np.float32)
    if vector.size != manifest.dim:
        raise IndexStaleError(
            f"query vector is {vector.size}d but the index is {manifest.dim}d; "
            "check embed.model and embed.dim"
        )
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _rank(
    dense: np.ndarray,
    bm25: np.ndarray,
    mode: str,
) -> tuple[list[int], np.ndarray]:
    if mode == "dense":
        return list(np.argsort(-dense, kind="stable")), dense
    if mode == "bm25":
        return list(np.argsort(-bm25, kind="stable")), bm25
    fused = np.zeros_like(dense)
    for rank, index in enumerate(np.argsort(-dense, kind="stable"), start=1):
        fused[index] += 1.0 / (RRF_K + rank)
    for rank, index in enumerate(np.argsort(-bm25, kind="stable"), start=1):
        fused[index] += 1.0 / (RRF_K + rank)
    return list(np.argsort(-fused, kind="stable")), fused


def _bm25_scores(texts: list[str], query: str) -> np.ndarray:
    documents = [_TOKEN.findall(text.lower()) for text in texts]
    count = len(documents)
    if count == 0:
        return np.zeros(0, dtype=np.float32)
    average_length = sum(len(document) for document in documents) / count
    terms = _TOKEN.findall(query.lower())
    document_frequency: Counter[str] = Counter()
    for document in documents:
        for term in set(document):
            document_frequency[term] += 1
    k1, b = 1.5, 0.75
    scores = np.zeros(count, dtype=np.float32)
    for index, document in enumerate(documents):
        length = len(document)
        frequencies = Counter(document)
        for term in terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            idf = math.log(
                1 + (count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)
            )
            denominator = frequency + k1 * (1 - b + b * (length / average_length))
            scores[index] += idf * (frequency * (k1 + 1)) / denominator
    return scores
