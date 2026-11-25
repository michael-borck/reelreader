from __future__ import annotations

import json

import numpy as np
import pytest

from reelreader.embed import (
    EmbeddingError,
    IndexStaleError,
    embed_video,
    ensure_embeddings,
    load_manifest,
    manifest_path,
)
from reelreader.library import Library, SegmentRecord


class FakeEmbedder:
    name = "fake"

    def __init__(self, *, model: str = "fake-model", dim: int = 3, vectors=None):
        self.model = model
        self.dim = dim
        self._vectors = vectors
        self.calls = 0

    def embed_texts(self, texts):
        self.calls += 1
        if self._vectors is not None:
            return [self._vectors[text] for text in texts]
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in text.lower().split():
                vector[sum(map(ord, token)) % self.dim] += 1.0
            if not any(vector):
                vector[0] = 1.0
            vectors.append(vector)
        return vectors


@pytest.fixture
def library(tmp_path) -> Library:
    return Library.open(tmp_path / "lib")


def add_video(library: Library, texts=("first", "second"), *, sha="a" * 64, title="lecture"):
    path = library.root / f"{sha[:4]}.mp4"
    path.write_bytes(b"x")
    video, _ = library.add_video(path=path, sha256=sha, title=title, duration=60.0, has_audio=True)
    library.replace_segments(
        video.id,
        [
            SegmentRecord(id=f"{sha[:4]}-{index}", start=index * 10, end=index * 10 + 10, text=text)
            for index, text in enumerate(texts)
        ],
    )
    return video


def test_embeds_and_writes_a_normalised_matrix(library):
    video = add_video(library)
    manifest = embed_video(library, video, FakeEmbedder(dim=3))

    path = library.root / "index" / video.id / "text.npy"
    assert path.is_file()
    matrix = np.load(path)
    assert matrix.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), [1.0, 1.0], rtol=1e-5)

    assert manifest.model == "fake-model"
    assert manifest.dim == 3
    assert manifest.videos[video.id]["count"] == 2
    assert manifest.videos[video.id]["segment_ids"] == ["aaaa-0", "aaaa-1"]

    on_disk = json.loads(manifest_path(library).read_text())
    assert on_disk["format"] == 1
    assert on_disk["videos"][video.id]["file"] == f"{video.id}/text.npy"


def test_second_run_skips(library):
    video = add_video(library)
    embedder = FakeEmbedder()
    ensure_embeddings(library, [video], embedder)
    summary = ensure_embeddings(library, [video], embedder)
    assert summary.embedded == ()
    assert summary.skipped == (video.id,)
    assert embedder.calls == 1


def test_force_re_embeds(library):
    video = add_video(library)
    embedder = FakeEmbedder()
    ensure_embeddings(library, [video], embedder)
    summary = ensure_embeddings(library, [video], embedder, force=True)
    assert summary.embedded == (video.id,)
    assert embedder.calls == 2


def test_model_change_requires_rebuild(library):
    video = add_video(library)
    ensure_embeddings(library, [video], FakeEmbedder(model="first"))
    with pytest.raises(IndexStaleError, match="rebuild"):
        ensure_embeddings(library, [video], FakeEmbedder(model="second"))


def test_model_change_with_force_rebuilds_everything(library):
    video = add_video(library)
    ensure_embeddings(library, [video], FakeEmbedder(model="first"))
    summary = ensure_embeddings(library, [video], FakeEmbedder(model="second", dim=2), force=True)
    assert summary.embedded == (video.id,)
    assert load_manifest(library).model == "second"
    assert load_manifest(library).dim == 2


def test_dimension_mismatch_is_reported(library):
    video = add_video(library)

    class ShortEmbedder(FakeEmbedder):
        def __init__(self):
            super().__init__(dim=3)

        def embed_texts(self, texts):
            return [[1.0, 0.0] for _ in texts]

    with pytest.raises(EmbeddingError, match="embed.dim"):
        embed_video(library, video, ShortEmbedder())


def test_video_without_segments_is_actionable(library):
    path = library.root / "empty.mp4"
    path.write_bytes(b"x")
    video, _ = library.add_video(
        path=path, sha256="b" * 64, title="empty", duration=10.0, has_audio=True
    )
    with pytest.raises(EmbeddingError, match="no segments"):
        embed_video(library, video, FakeEmbedder())


def test_invalid_manifest_is_reported(library):
    path = manifest_path(library)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format": 99, "model": "x", "dim": 3}))
    with pytest.raises(EmbeddingError, match="unsupported index format"):
        load_manifest(library)


def test_embedded_videos_are_independent(library):
    first = add_video(library, sha="a" * 64)
    second = add_video(library, sha="b" * 64, title="second")
    embedder = FakeEmbedder()
    ensure_embeddings(library, [first], embedder)
    summary = ensure_embeddings(library, [second], embedder)
    assert summary.embedded == (second.id,)
    manifest = load_manifest(library)
    assert set(manifest.videos) == {first.id, second.id}
    assert (library.root / "index" / second.id / "text.npy").is_file()
