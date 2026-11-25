from __future__ import annotations

import importlib.util

import pytest

from reelreader.transcribe import (
    FasterWhisperTranscriber,
    Sentence,
    Transcript,
    TranscriptionError,
    Word,
    build_sentences,
    empty_transcript,
    load_transcript,
    save_transcript,
)


def words(*items):
    return [Word(start=start, end=end, text=text) for start, end, text in items]


def test_splits_on_punctuation():
    sentences = build_sentences(
        words(
            (0.0, 0.3, "Hello"),
            (0.3, 0.5, "there."),
            (0.6, 0.9, "Next"),
            (0.9, 1.1, "point!"),
        )
    )
    assert [sentence.text for sentence in sentences] == ["Hello there.", "Next point!"]
    assert sentences[1].start == 0.6


def test_splits_on_a_long_gap():
    sentences = build_sentences(
        words((0.0, 0.4, "Before."), (2.0, 2.4, "After.")),
        gap_seconds=0.8,
    )
    assert len(sentences) == 2


def test_splits_on_length():
    many = [Word(start=index * 0.1, end=index * 0.1 + 0.1, text="word") for index in range(80)]
    sentences = build_sentences(many, max_chars=60)
    assert len(sentences) > 1
    assert all(len(sentence.text) <= 80 for sentence in sentences)


def test_ignores_empty_words():
    sentences = build_sentences(words((0.0, 0.2, " "), (0.2, 0.4, "Hi.")))
    assert len(sentences) == 1
    assert sentences[0].text == "Hi."


def test_transcript_roundtrip(tmp_path):
    transcript = Transcript(
        language="en",
        engine="test",
        duration=12.5,
        words=(Word(0.0, 0.5, "Hi."),),
        sentences=(Sentence(0.0, 0.5, "Hi."),),
    )
    path = tmp_path / "transcript.json"
    save_transcript(path, transcript)
    loaded = load_transcript(path)
    assert loaded == transcript
    assert loaded.to_dict()["format"] == 1


def test_invalid_transcript_is_explained(tmp_path):
    path = tmp_path / "transcript.json"
    path.write_text("{not json")
    with pytest.raises(TranscriptionError, match="not valid JSON"):
        load_transcript(path)


def test_missing_transcript():
    from pathlib import Path

    with pytest.raises(TranscriptionError, match="does not exist"):
        load_transcript(Path("nope.json"))


def test_empty_transcript_for_video_without_speech():
    transcript = empty_transcript(42.0)
    assert transcript.duration == 42.0
    assert transcript.words == ()
    assert transcript.sentences == ()
    assert transcript.engine == "none"


@pytest.mark.skipif(
    importlib.util.find_spec("faster_whisper") is not None,
    reason="faster-whisper is installed",
)
def test_missing_asr_dependency_is_actionable(tmp_path):
    transcriber = FasterWhisperTranscriber("tiny")
    with pytest.raises(TranscriptionError, match="reelreader\\[asr\\]"):
        transcriber.transcribe(tmp_path / "audio.wav")
