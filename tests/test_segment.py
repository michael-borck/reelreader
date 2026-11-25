from __future__ import annotations

from reelreader.segment import Segment, build_segments, load_segments, save_segments
from reelreader.transcribe import Sentence


def sentences(*items):
    return [Sentence(start=start, end=end, text=text) for start, end, text in items]


def test_no_sentences_still_covers_the_video():
    segments = build_segments([], duration=90.0)
    assert len(segments) == 1
    assert (segments[0].start, segments[0].end) == (0.0, 90.0)
    assert segments[0].text == ""
    assert segments[0].keyframe_at == 45.0


def test_no_duration_and_no_sentences_produce_nothing():
    assert build_segments([], duration=0.0) == []


def test_closes_after_a_long_enough_gap():
    segments = build_segments(
        sentences(
            (0.0, 15.0, "First part."),
            (16.5, 31.0, "Second part."),
            (33.0, 48.0, "Third part."),
        ),
        duration=48.0,
    )
    assert [segment.end for segment in segments] == [31.0, 48.0]
    assert segments[0].text == "First part. Second part."
    assert [segment.index for segment in segments] == [1, 2]


def test_closes_on_a_scene_cut():
    segments = build_segments(
        sentences((0.0, 18.0, "Before the slide."), (18.5, 30.0, "After the slide.")),
        cuts=[18.2],
        duration=30.0,
    )
    assert len(segments) == 2
    assert segments[0].end == 18.0
    assert segments[1].start == 18.5


def test_closes_at_the_maximum_span():
    segments = build_segments(
        sentences(
            (0.0, 20.0, "A."),
            (20.1, 40.0, "B."),
            (40.1, 60.0, "C."),
            (60.1, 80.0, "D."),
        ),
        duration=80.0,
        max_seconds=42.0,
    )
    assert all(segment.duration <= 42.5 for segment in segments)
    assert len(segments) >= 2


def test_merges_a_short_tail():
    segments = build_segments(
        sentences(
            (0.0, 20.0, "Long."),
            (21.5, 40.0, "Also long."),
            (41.5, 45.0, "Tiny tail."),
        ),
        duration=45.0,
    )
    assert len(segments) == 2
    assert segments[-1].end == 45.0
    assert "Tiny tail." in segments[-1].text


def test_clamps_to_duration():
    segments = build_segments(
        sentences((0.0, 10.0, "A."), (10.5, 99.0, "B.")),
        duration=20.0,
    )
    assert all(segment.end <= 20.0 for segment in segments)


def test_ids_are_unique_and_keyframes_centred():
    segments = build_segments(
        sentences((0.0, 20.0, "A."), (22.0, 40.0, "B."), (42.0, 60.0, "C.")),
        duration=60.0,
    )
    assert len({segment.id for segment in segments}) == len(segments)
    for segment in segments:
        assert segment.keyframe_at == (segment.start + segment.end) / 2


def test_segments_roundtrip(tmp_path):
    original = build_segments(sentences((0.0, 30.0, "Hello.")), duration=30.0)
    path = tmp_path / "segments.json"
    save_segments(path, video_id="v1", source_sha256="a" * 64, segments=original)
    loaded = load_segments(path)
    assert loaded == original


def test_segment_record_shape():
    segment = Segment(index=1, id="x", start=0.0, end=5.0, text="hi", keyframe_at=2.5)
    assert segment.duration == 5.0
    assert segment.to_dict()["artifacts"] == {}
