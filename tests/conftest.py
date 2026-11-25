from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    for name in list(os.environ):
        if name.startswith("REELREADER_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path
