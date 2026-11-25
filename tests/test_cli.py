from __future__ import annotations

import json

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
    from pathlib import Path

    result = invoke("init")
    assert result.exit_code == 0
    assert Path("reelreader.toml").is_file()


def test_planned_command_is_honest():
    result = invoke("ask", "what is a p-value?")
    assert result.exit_code == 2
    assert "Not in v0.0.1" in result.stdout
