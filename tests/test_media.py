from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from reelreader.media import ProbeError, parse_probe, probe

SAMPLE = {
    "format": {"duration": "125.5"},
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1280,
            "height": 720,
            "avg_frame_rate": "30000/1001",
        },
        {"codec_type": "audio", "codec_name": "aac"},
    ],
}


def test_parses_format_and_streams():
    info = parse_probe(SAMPLE, Path("lecture.mp4"))
    assert info.duration == 125.5
    assert (info.width, info.height) == (1280, 720)
    assert info.fps == 29.97
    assert info.has_audio is True
    assert info.video_codec == "h264"
    assert info.audio_codec == "aac"


def test_ignores_cover_art():
    data = {
        "streams": [
            {
                "codec_type": "video",
                "disposition": {"attached_pic": 1},
                "width": 500,
                "height": 500,
            }
        ]
    }
    with pytest.raises(ProbeError, match="no video stream"):
        parse_probe(data, Path("audio.m4a"))


def test_rejects_audio_only():
    data = {"streams": [{"codec_type": "audio", "codec_name": "mp3"}]}
    with pytest.raises(ProbeError, match="no video stream"):
        parse_probe(data, Path("audio.mp3"))


def test_missing_file():
    with pytest.raises(ProbeError, match="does not exist"):
        probe(Path("does-not-exist.mp4"))


def test_missing_ffprobe(tmp_path, monkeypatch):
    source = tmp_path / "a.mp4"
    source.write_bytes(b"x")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(ProbeError, match="ffprobe not found"):
        probe(source)


def test_reports_ffprobe_failure(tmp_path):
    source = tmp_path / "a.mp4"
    source.write_bytes(b"x")

    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="bad container")

    with pytest.raises(ProbeError, match="bad container"):
        probe(source, ffprobe="/fake/ffprobe", run=run)


def test_reports_invalid_json(tmp_path):
    source = tmp_path / "a.mp4"
    source.write_bytes(b"x")

    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="{nope", stderr="")

    with pytest.raises(ProbeError, match="invalid JSON"):
        probe(source, ffprobe="/fake/ffprobe", run=run)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_probes_a_real_video(tmp_path):
    target = tmp_path / "real.mp4"
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=64x48:r=10:d=0.6",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=8000:cl=mono",
        "-shortest",
        "-c:v",
        "mpeg4",
        "-c:a",
        "aac",
        str(target),
    ]
    subprocess.run(command, check=True, capture_output=True)
    info = probe(target)
    assert (info.width, info.height) == (64, 48)
    assert info.has_audio is True
    assert 0.4 < info.duration < 1.0
