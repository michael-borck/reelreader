from __future__ import annotations

import sqlite3

import pytest

from reelreader.library import Library, LibraryError
from reelreader.media import MediaInfo

INFO = MediaInfo(
    duration=90.0,
    width=640,
    height=360,
    fps=25.0,
    has_audio=True,
    video_codec="h264",
    audio_codec="aac",
)


def add(library: Library, name: str = "a.mp4", sha: str = "a" * 64, **kwargs):
    return library.add_video(
        path=library.root / name,
        sha256=sha,
        title=name,
        duration=INFO.duration,
        has_audio=kwargs.pop("has_audio", True),
        **kwargs,
    )


def test_open_creates_folders_and_schema(tmp_path):
    library = Library.open(tmp_path / "lib")
    assert library.db_path.is_file()
    assert (library.root / "videos").is_dir()
    assert (library.root / "index").is_dir()
    with sqlite3.connect(library.db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1


def test_open_is_idempotent(tmp_path):
    root = tmp_path / "lib"
    Library.open(root)
    Library.open(root)


def test_add_and_get(tmp_path):
    library = Library.open(tmp_path / "lib")
    video, created = add(library)
    assert created is True
    assert video.status == "ingested"
    assert library.get_video(video.id).title == "a.mp4"


def test_no_audio_sets_status(tmp_path):
    library = Library.open(tmp_path / "lib")
    video, _ = add(library, has_audio=False)
    assert video.status == "no_audio"


def test_deduplicates_by_sha(tmp_path):
    library = Library.open(tmp_path / "lib")
    first, created = add(library)
    second, created_again = add(library, name="copy.mp4")
    assert created is True
    assert created_again is False
    assert second.id == first.id
    assert len(library.list_videos()) == 1


def test_rejects_unknown_status(tmp_path):
    library = Library.open(tmp_path / "lib")
    with pytest.raises(LibraryError, match="unknown status"):
        add(library, status="cinema")


def test_unknown_video_raises(tmp_path):
    library = Library.open(tmp_path / "lib")
    with pytest.raises(LibraryError, match="no video"):
        library.get_video("missing")


def test_set_status_and_delete(tmp_path):
    library = Library.open(tmp_path / "lib")
    video, _ = add(library)
    library.set_status(video.id, "ready")
    assert library.get_video(video.id).status == "ready"
    library.delete_video(video.id)
    assert library.list_videos() == []


def test_persists_across_reopen(tmp_path):
    root = tmp_path / "lib"
    video, _ = add(Library.open(root))
    reopened = Library.open(root)
    assert reopened.get_video(video.id).sha256 == "a" * 64


def test_job_lifecycle(tmp_path):
    library = Library.open(tmp_path / "lib")
    video, _ = add(library)
    job = library.start_job(video.id, "transcribe")
    assert job.state == "running"
    assert job.progress == 0.0

    updated = library.update_job(job.id, progress=0.5, message="halfway")
    assert updated.progress == 0.5
    assert updated.message == "halfway"

    finished = library.finish_job(job.id, message="done")
    assert finished.state == "done"
    assert finished.progress == 1.0
    assert library.latest_job(video.id).id == job.id


def test_failed_job_records_error(tmp_path):
    library = Library.open(tmp_path / "lib")
    video, _ = add(library)
    job = library.start_job(video.id, "embed")
    failed = library.fail_job(job.id, "model missing")
    assert failed.state == "failed"
    assert failed.error == "model missing"


def test_latest_job_is_none_for_new_video(tmp_path):
    library = Library.open(tmp_path / "lib")
    video, _ = add(library)
    assert library.latest_job(video.id) is None
