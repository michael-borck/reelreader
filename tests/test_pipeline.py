from __future__ import annotations

import shutil
import subprocess

import pytest

from reelreader.library import Library
from reelreader.pipeline import PipelineError, index_video
from reelreader.segment import load_segments
from reelreader.transcribe import Sentence, Transcript, TranscriptionError

FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")

TRANSCRIPT = Transcript(
    language="en",
    engine="fake",
    duration=1.5,
    words=(),
    sentences=(Sentence(0.0, 0.6, "Hello there."), Sentence(0.7, 1.4, "A small test.")),
)


class FakeTranscriber:
    name = "fake"

    def __init__(self, transcript: Transcript | None = None, error: Exception | None = None):
        self.transcript = transcript or TRANSCRIPT
        self.error = error
        self.calls = 0

    def transcribe(self, audio, *, duration=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.transcript


@pytest.fixture
def video_file(tmp_path):
    target = tmp_path / "lecture.mp4"
    command = [
        str(FFMPEG),
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x180:r=10:d=1.5",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=16000:cl=mono",
        "-shortest",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        str(target),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return target


@pytest.fixture
def library(tmp_path) -> Library:
    return Library.open(tmp_path / "lib")


def add_video(library: Library, path, *, sha: str = "a" * 64, has_audio: bool = True):
    video, _ = library.add_video(
        path=path,
        sha256=sha,
        title="lecture",
        duration=1.5,
        width=320,
        height=180,
        has_audio=has_audio,
    )
    return video


def test_full_run_writes_everything(library, video_file):
    video = add_video(library, video_file)
    result = index_video(library, video, transcriber=FakeTranscriber(), scene_cuts=[])
    assert result.segment_count == 1
    assert result.ran == ("transcode", "transcribe", "segment", "artifacts")

    out = library.root / "videos" / video.id
    for name in ("proxy.mp4", "audio.wav", "transcript.json", "segments.json"):
        assert (out / name).is_file(), name

    segments = load_segments(out / "segments.json")
    assert len(segments) == 1
    for relative in segments[0].artifacts.values():
        assert (out / relative).is_file(), relative

    assert library.get_video(video.id).status == "ready"
    assert len(library.list_segments(video.id)) == 1
    job = library.latest_job(video.id)
    assert job is not None
    assert job.state == "done"


def test_second_run_is_a_noop(library, video_file):
    video = add_video(library, video_file)
    index_video(library, video, transcriber=FakeTranscriber(), scene_cuts=[])
    again = index_video(library, video, transcriber=FakeTranscriber(), scene_cuts=[])
    assert again.ran == ()
    assert again.skipped == ("transcode", "transcribe", "segment", "artifacts")
    assert len(library.list_segments(video.id)) == 1


def test_resumes_after_a_missing_transcript(library, video_file):
    video = add_video(library, video_file)
    index_video(library, video, transcriber=FakeTranscriber(), scene_cuts=[])
    out = library.root / "videos" / video.id
    (out / "transcript.json").unlink()

    transcriber = FakeTranscriber()
    resumed = index_video(library, video, transcriber=transcriber, scene_cuts=[])
    assert resumed.ran == ("transcribe",)
    assert "segment" in resumed.skipped
    assert transcriber.calls == 1


def test_rebuild_ignores_existing_outputs(library, video_file):
    video = add_video(library, video_file)
    index_video(library, video, transcriber=FakeTranscriber(), scene_cuts=[])
    forced = index_video(library, video, transcriber=FakeTranscriber(), scene_cuts=[], force=True)
    assert forced.ran == ("transcode", "transcribe", "segment", "artifacts")


def test_transcription_failure_marks_the_video_failed(library, video_file):
    video = add_video(library, video_file)
    with pytest.raises(PipelineError, match="boom"):
        index_video(
            library,
            video,
            transcriber=FakeTranscriber(error=TranscriptionError("boom")),
            scene_cuts=[],
        )
    assert library.get_video(video.id).status == "failed"
    job = library.latest_job(video.id)
    assert job is not None
    assert job.state == "failed"
    assert job.error == "boom"


def test_video_without_audio_skips_speech_and_wav(library, video_file):
    video = add_video(library, video_file, sha="b" * 64, has_audio=False)
    transcriber = FakeTranscriber()
    result = index_video(library, video, transcriber=transcriber, scene_cuts=[])
    assert transcriber.calls == 0
    assert result.segment_count == 1

    out = library.root / "videos" / video.id
    assert not (out / "audio.wav").exists()
    segments = load_segments(out / "segments.json")
    assert "audio" not in segments[0].artifacts
    assert segments[0].text == ""


def test_missing_source_is_reported(library, tmp_path):
    video = add_video(library, tmp_path / "gone.mp4", sha="c" * 64)
    with pytest.raises(PipelineError, match="source file is missing"):
        index_video(library, video, transcriber=FakeTranscriber(), scene_cuts=[])
    assert library.get_video(video.id).status == "failed"


def test_interrupt_marks_the_job_and_resumes(library, video_file):
    video = add_video(library, video_file)

    class Interrupting:
        name = "interrupting"

        def transcribe(self, audio, *, duration=None):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        index_video(library, video, transcriber=Interrupting(), scene_cuts=[])

    assert library.get_video(video.id).status == "interrupted"
    job = library.latest_job(video.id)
    assert job is not None
    assert job.state == "cancelled"

    resumed = index_video(library, video, transcriber=FakeTranscriber(), scene_cuts=[])
    assert "transcribe" in resumed.ran
    assert library.get_video(video.id).status == "ready"
