"""The library database: videos, segments, jobs, and chats.

One folder is one library. The database is derived index metadata plus the
user's own chat history; the media and transcript files remain the source of
truth, so a deleted database can be rebuilt by re-ingesting.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from reelreader.ids import new_id

SCHEMA_VERSION = 1

STATUSES = ("ingested", "indexing", "ready", "failed", "no_audio", "missing", "interrupted")
JOB_STATES = ("queued", "running", "done", "failed", "cancelled")


class LibraryError(RuntimeError):
    """The library could not be opened or written."""


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class Video:
    id: str
    path: Path
    sha256: str
    title: str
    duration: float
    width: int | None
    height: int | None
    has_audio: bool
    status: str
    added_at: str
    source: str

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    @property
    def duration_label(self) -> str:
        total = int(round(self.duration))
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"


@dataclass(frozen=True, slots=True)
class SegmentRecord:
    id: str
    start: float
    end: float
    text: str
    keyframe_at: float | None = None


@dataclass(frozen=True, slots=True)
class Job:
    id: str
    video_id: str
    stage: str
    state: str
    progress: float
    message: str | None
    error: str | None
    created_at: str
    updated_at: str


SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS videos (
    id          TEXT PRIMARY KEY,
    path        TEXT NOT NULL,
    sha256      TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL,
    duration    REAL NOT NULL DEFAULT 0,
    width       INTEGER,
    height      INTEGER,
    has_audio   INTEGER NOT NULL DEFAULT 1,
    status      TEXT NOT NULL DEFAULT 'ingested',
    added_at    TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'file'
);

CREATE TABLE IF NOT EXISTS segments (
    id          TEXT PRIMARY KEY,
    video_id    TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    start       REAL NOT NULL,
    end         REAL NOT NULL,
    text        TEXT NOT NULL,
    keyframe_at REAL
);

CREATE INDEX IF NOT EXISTS segments_by_video ON segments(video_id, start);

CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    video_id    TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    stage       TEXT NOT NULL,
    state       TEXT NOT NULL,
    progress    REAL NOT NULL DEFAULT 0,
    message     TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS jobs_by_video ON jobs(video_id, created_at DESC);

CREATE TABLE IF NOT EXISTS chats (
    id          TEXT PRIMARY KEY,
    video_id    TEXT REFERENCES videos(id) ON DELETE SET NULL,
    question    TEXT NOT NULL,
    answer      TEXT,
    citations   TEXT,
    created_at  TEXT NOT NULL
);
"""


class Library:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.db_path = root / "library.db"

    @classmethod
    def open(cls, root: Path | str) -> Library:
        root_path = Path(root).expanduser().resolve()
        try:
            root_path.mkdir(parents=True, exist_ok=True)
            (root_path / "videos").mkdir(exist_ok=True)
            (root_path / "index").mkdir(exist_ok=True)
        except OSError as exc:
            raise LibraryError(f"cannot create library at {root_path}: {exc}") from exc
        library = cls(root_path)
        try:
            with library._connect() as conn:
                _migrate(conn)
        except sqlite3.Error as exc:
            raise LibraryError(f"cannot open {library.db_path}: {exc}") from exc
        return library

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def add_video(
        self,
        *,
        path: Path,
        sha256: str,
        title: str,
        duration: float,
        width: int | None = None,
        height: int | None = None,
        has_audio: bool = True,
        source: str = "file",
        status: str | None = None,
        video_id: str | None = None,
    ) -> tuple[Video, bool]:
        resolved_status = status or ("ingested" if has_audio else "no_audio")
        _require(resolved_status in STATUSES, f"unknown status: {resolved_status}")
        identifier = video_id or new_id()
        added_at = utcnow()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO videos
                        (id, path, sha256, title, duration, width, height,
                         has_audio, status, added_at, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identifier,
                        str(path),
                        sha256,
                        title,
                        duration,
                        width,
                        height,
                        int(has_audio),
                        resolved_status,
                        added_at,
                        source,
                    ),
                )
        except sqlite3.IntegrityError:
            existing = self.find_by_sha(sha256)
            if existing is None:
                raise
            return existing, False
        except sqlite3.Error as exc:
            raise LibraryError(f"cannot add video: {exc}") from exc
        video = self.get_video(identifier)
        return video, True

    def get_video(self, video_id: str) -> Video:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
        if row is None:
            raise LibraryError(f"no video with id {video_id}")
        return _row_to_video(row)

    def find_by_sha(self, sha256: str) -> Video | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM videos WHERE sha256 = ?", (sha256,)).fetchone()
        return _row_to_video(row) if row else None

    def list_videos(self) -> list[Video]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM videos ORDER BY added_at DESC, id DESC").fetchall()
        return [_row_to_video(row) for row in rows]

    def set_status(self, video_id: str, status: str) -> None:
        _require(status in STATUSES, f"unknown status: {status}")
        with self._connect() as conn:
            conn.execute("UPDATE videos SET status = ? WHERE id = ?", (status, video_id))

    def delete_video(self, video_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))

    def replace_segments(self, video_id: str, segments: list[SegmentRecord]) -> None:
        rows = [
            (segment.id, video_id, segment.start, segment.end, segment.text, segment.keyframe_at)
            for segment in segments
        ]
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM segments WHERE video_id = ?", (video_id,))
                conn.executemany(
                    """
                    INSERT INTO segments (id, video_id, start, end, text, keyframe_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        except sqlite3.Error as exc:
            raise LibraryError(f"cannot save segments: {exc}") from exc

    def list_segments(self, video_id: str) -> list[SegmentRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM segments WHERE video_id = ? ORDER BY start", (video_id,)
            ).fetchall()
        return [
            SegmentRecord(
                id=row["id"],
                start=row["start"],
                end=row["end"],
                text=row["text"],
                keyframe_at=row["keyframe_at"],
            )
            for row in rows
        ]

    def start_job(self, video_id: str, stage: str) -> Job:
        job_id = new_id()
        now = utcnow()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO jobs
                        (id, video_id, stage, state, progress, created_at, updated_at)
                    VALUES (?, ?, ?, 'running', 0, ?, ?)
                    """,
                    (job_id, video_id, stage, now, now),
                )
        except sqlite3.Error as exc:
            raise LibraryError(f"cannot start job: {exc}") from exc
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Job:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise LibraryError(f"no job with id {job_id}")
        return _row_to_job(row)

    def update_job(
        self,
        job_id: str,
        *,
        progress: float | None = None,
        message: str | None = None,
        state: str | None = None,
        error: str | None = None,
    ) -> Job:
        if state is not None:
            _require(state in JOB_STATES, f"unknown job state: {state}")
        fields: list[str] = ["updated_at = ?"]
        values: list[object] = [utcnow()]
        for column, value in (
            ("progress", progress),
            ("message", message),
            ("state", state),
            ("error", error),
        ):
            if value is not None:
                fields.append(f"{column} = ?")
                values.append(value)
        values.append(job_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", values)
        return self.get_job(job_id)

    def finish_job(self, job_id: str, message: str | None = None) -> Job:
        return self.update_job(job_id, state="done", progress=1.0, message=message)

    def fail_job(self, job_id: str, error: str) -> Job:
        return self.update_job(job_id, state="failed", error=error)

    def latest_job(self, video_id: str) -> Job | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE video_id = ? ORDER BY created_at DESC, id DESC LIMIT 1",
                (video_id,),
            ).fetchone()
        return _row_to_job(row) if row else None


def _migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version < 1:
        conn.executescript(SCHEMA_V1)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def _row_to_video(row: sqlite3.Row) -> Video:
    return Video(
        id=row["id"],
        path=Path(row["path"]),
        sha256=row["sha256"],
        title=row["title"],
        duration=row["duration"],
        width=row["width"],
        height=row["height"],
        has_audio=bool(row["has_audio"]),
        status=row["status"],
        added_at=row["added_at"],
        source=row["source"],
    )


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        video_id=row["video_id"],
        stage=row["stage"],
        state=row["state"],
        progress=row["progress"],
        message=row["message"],
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LibraryError(message)
