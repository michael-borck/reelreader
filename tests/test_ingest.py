from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from reelreader import ingest
from reelreader.ingest import IngestError, ingest_file, ingest_path, ingest_url
from reelreader.library import Library
from reelreader.media import MediaInfo

WITH_AUDIO = MediaInfo(
    duration=90.0,
    width=640,
    height=360,
    fps=25.0,
    has_audio=True,
    video_codec="h264",
    audio_codec="aac",
)
NO_AUDIO = MediaInfo(
    duration=90.0,
    width=640,
    height=360,
    fps=25.0,
    has_audio=False,
    video_codec="h264",
    audio_codec=None,
)


@pytest.fixture
def library(tmp_path) -> Library:
    return Library.open(tmp_path / "lib")


@pytest.fixture
def video(tmp_path) -> Path:
    path = tmp_path / "lecture.mp4"
    path.write_bytes(b"not really a video")
    return path


def test_missing_source(library, tmp_path):
    with pytest.raises(IngestError, match="does not exist"):
        ingest_path(library, tmp_path / "nope.mp4")


def test_unsupported_extension(library, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(IngestError, match="not a video file"):
        ingest_path(library, path)


def test_references_the_original_by_default(library, video):
    result = ingest_file(library, video, probe_fn=lambda _: WITH_AUDIO)
    assert result.created is True
    assert result.video.path == video.resolve()
    assert result.video.title == "lecture"
    assert result.video.status == "ingested"


def test_copy_brings_the_file_into_the_library(library, video):
    result = ingest_file(library, video, copy=True, probe_fn=lambda _: WITH_AUDIO)
    stored = result.video.path
    assert stored.parent == library.root / "videos" / result.video.id
    assert stored.name == "source.mp4"
    assert stored.read_bytes() == video.read_bytes()


def test_second_add_is_a_noop(library, video):
    first = ingest_file(library, video, probe_fn=lambda _: WITH_AUDIO)
    second = ingest_file(library, video, probe_fn=lambda _: WITH_AUDIO)
    assert first.created is True
    assert second.created is False
    assert second.skipped == "already in library"
    assert second.video.id == first.video.id


def test_no_audio_is_accepted(library, video):
    result = ingest_file(library, video, probe_fn=lambda _: NO_AUDIO)
    assert result.video.status == "no_audio"
    assert result.video.has_audio is False


def test_probe_failure_becomes_ingest_error(library, video):
    def boom(_: Path) -> MediaInfo:
        raise RuntimeError("no video stream")

    with pytest.raises(IngestError, match="no video stream"):
        ingest_file(library, video, probe_fn=boom)


def test_folder_scans_nested_videos(library, tmp_path):
    folder = tmp_path / "week3"
    (folder / "nested").mkdir(parents=True)
    for name in ("nested/a.mp4", "b.mkv", "notes.txt", ".hidden.mp4"):
        target = folder / name
        target.write_bytes(name.encode())
    results = ingest_path(library, folder, probe_fn=lambda _: WITH_AUDIO)
    assert len(results) == 2
    assert all(result.created for result in results)


def test_empty_folder_reports_clearly(library, tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    with pytest.raises(IngestError, match="no video files found"):
        ingest_path(library, folder)


def test_url_without_ytdlp(library, monkeypatch):
    monkeypatch.setattr(ingest.shutil, "which", lambda name: None)
    with pytest.raises(IngestError, match="yt-dlp"):
        ingest_url(library, "https://example.com/lecture")


def test_url_download_then_ingest(library, monkeypatch):
    monkeypatch.setattr(ingest.shutil, "which", lambda name: "/fake/yt-dlp")

    def fake_run(command, **kwargs):
        template = command[command.index("-o") + 1]
        target = Path(template.replace("%(ext)s", "mp4"))
        target.write_bytes(b"downloaded video")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(ingest.subprocess, "run", fake_run)
    result = ingest_url(
        library, "https://example.com/lecture", title="Week 3", probe_fn=lambda _: WITH_AUDIO
    )
    assert result.created is True
    assert result.video.title == "Week 3"
    assert result.video.source == "url"
    assert result.video.path.is_file()
    assert library.root in result.video.path.parents


def test_url_failure_is_actionable(library, monkeypatch):
    monkeypatch.setattr(ingest.shutil, "which", lambda name: "/fake/yt-dlp")

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="HTTP 403")

    monkeypatch.setattr(ingest.subprocess, "run", fake_run)
    with pytest.raises(IngestError, match="HTTP 403"):
        ingest_url(library, "https://example.com/lecture")
