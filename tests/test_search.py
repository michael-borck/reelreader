from __future__ import annotations

import pytest

from reelreader.embed import IndexStaleError, IndexUnavailableError, ensure_embeddings
from reelreader.library import Library, SegmentRecord
from reelreader.search import SearchError, format_time, search_library


class HashEmbedder:
    name = "hash"

    def __init__(self, *, model: str = "hash-model", dim: int = 64):
        self.model = model
        self.dim = dim

    def embed_texts(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in text.lower().split():
                vector[sum(map(ord, token)) % self.dim] += 1.0
            if not any(vector):
                vector[0] = 1.0
            vectors.append(vector)
        return vectors


TEXTS = [
    "A p value measures how likely the data are under the null hypothesis.",
    "If the p value is below alpha we reject the null hypothesis.",
    "We will take a short coffee break before the next topic.",
]


@pytest.fixture
def library(tmp_path) -> Library:
    lib = Library.open(tmp_path / "lib")
    path = lib.root / "lecture.mp4"
    path.write_bytes(b"x")
    video, _ = lib.add_video(path=path, sha256="a" * 64, title="Statistics week 3", duration=120.0)
    lib.replace_segments(
        video.id,
        [
            SegmentRecord(id=f"s{index}", start=index * 20, end=index * 20 + 20, text=text)
            for index, text in enumerate(TEXTS)
        ],
    )
    ensure_embeddings(lib, [video], HashEmbedder())
    return lib


def test_dense_finds_the_meaningful_moment(library):
    hits = search_library(library, "p value", embedder=HashEmbedder(), mode="dense")
    assert hits[0].text == TEXTS[0]
    assert hits[0].dense is not None


def test_bm25_finds_a_keyword(library):
    hits = search_library(library, "coffee", embedder=HashEmbedder(), mode="bm25")
    assert hits[0].text == TEXTS[2]
    assert hits[0].bm25 > 0


def test_hybrid_ranks_both_ways(library):
    hits = search_library(library, "coffee break", embedder=HashEmbedder(), mode="hybrid")
    assert hits[0].text == TEXTS[2]
    assert hits[0].score > 0


def test_limit_and_labels(library):
    hits = search_library(library, "hypothesis", embedder=HashEmbedder(), limit=2)
    assert len(hits) == 2
    assert hits[0].label == format_time(hits[0].start)


def test_empty_query_returns_nothing(library):
    assert search_library(library, "   ", embedder=HashEmbedder()) == []


def test_unknown_mode_is_rejected(library):
    with pytest.raises(SearchError, match="mode must be one of"):
        search_library(library, "p value", embedder=HashEmbedder(), mode="magic")


def test_video_filter(library):
    assert search_library(library, "p value", embedder=HashEmbedder(), video_id="nope") == []


def test_missing_index_is_actionable(tmp_path):
    library = Library.open(tmp_path / "empty")
    with pytest.raises(IndexUnavailableError, match="reelreader index"):
        search_library(library, "anything", embedder=HashEmbedder())


def test_model_mismatch_is_reported(library):
    with pytest.raises(IndexStaleError, match="rebuild"):
        search_library(library, "p value", embedder=HashEmbedder(model="other"))


def test_format_time_shapes():
    assert format_time(0) == "0:00"
    assert format_time(75) == "1:15"
    assert format_time(3675) == "1:01:15"
