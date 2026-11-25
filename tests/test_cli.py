from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from reelreader import __version__
from reelreader.cli import app

runner = CliRunner()


def invoke(*args: str):
    return runner.invoke(app, list(args))


def test_version_flag():
    result = invoke("--version")
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help_lists_the_surfaces():
    result = invoke("--help")
    assert result.exit_code == 0
    for command in ("doctor", "ask", "index", "serve", "export", "eval"):
        assert command in result.stdout


def test_doctor_runs_without_network():
    result = invoke("doctor")
    assert result.exit_code == 0
    assert "Reelreader doctor" in result.stdout


def test_config_prints_redacted_json():
    result = invoke("config")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["chat"]["provider"] == "openai"
    assert "api_key" not in payload["chat"]


def test_init_writes_config():
    result = invoke("init")
    assert result.exit_code == 0
    assert Path("reelreader.toml").is_file()


def test_add_then_list_then_skip(monkeypatch):
    from reelreader import ingest as ingest_module
    from reelreader.media import MediaInfo

    info = MediaInfo(
        duration=61.0,
        width=640,
        height=360,
        fps=25.0,
        has_audio=True,
        video_codec="h264",
        audio_codec="aac",
    )
    monkeypatch.setattr(ingest_module, "probe", lambda path: info)
    Path("lecture.mp4").write_bytes(b"pretend video")

    added = invoke("add", "lecture.mp4")
    assert added.exit_code == 0
    assert "Added" in added.stdout

    listed = invoke("list")
    assert listed.exit_code == 0
    assert "lecture" in listed.stdout

    again = invoke("add", "lecture.mp4")
    assert again.exit_code == 0
    assert "Skipped" in again.stdout


def test_add_json(monkeypatch):
    from reelreader import ingest as ingest_module
    from reelreader.media import MediaInfo

    info = MediaInfo(
        duration=61.0,
        width=640,
        height=360,
        fps=25.0,
        has_audio=True,
        video_codec="h264",
        audio_codec="aac",
    )
    monkeypatch.setattr(ingest_module, "probe", lambda path: info)
    Path("lecture.mp4").write_bytes(b"pretend video")

    result = invoke("add", "lecture.mp4", "--json")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload[0]["title"] == "lecture"
    assert payload[0]["created"] is True


def test_list_is_empty_by_default():
    result = invoke("list")
    assert result.exit_code == 0
    assert "No videos yet" in result.stdout


def test_planned_command_is_honest():
    result = invoke("ask", "what is a p-value?")
    assert result.exit_code == 2
    assert "Not in v0.0.1" in result.stdout
