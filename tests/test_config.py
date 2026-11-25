from __future__ import annotations

import pytest

from reelreader.config import ConfigError, Settings


def test_defaults():
    settings = Settings.load()
    assert settings.tier == "text"
    assert settings.chat.provider == "openai"
    assert settings.embed.provider == "local"
    assert settings.library.name == "Reelreader"
    assert settings.config_path is None


def test_reads_toml(isolated_env):
    (isolated_env / "reelreader.toml").write_text(
        """
library = "/tmp/reel"
tier = "visual"

[chat]
provider = "anthropic"
model = "claude-sonnet-4-5"
api_key_env = "MY_ANTHROPIC"

[embed]
provider = "openai"
model = "text-embedding-3-small"
dim = 1536
""",
        encoding="utf-8",
    )
    settings = Settings.load()
    assert settings.tier == "visual"
    assert settings.chat.provider == "anthropic"
    assert settings.chat.model == "claude-sonnet-4-5"
    assert settings.chat.api_key_env == "MY_ANTHROPIC"
    assert settings.embed.dim == 1536
    assert settings.config_path is not None


def test_env_overrides_file(isolated_env, monkeypatch):
    (isolated_env / "reelreader.toml").write_text(
        'tier = "text"\n[chat]\nmodel = "from-file"\n', encoding="utf-8"
    )
    monkeypatch.setenv("REELREADER_TIER", "full")
    monkeypatch.setenv("REELREADER_CHAT_MODEL", "from-env")
    settings = Settings.load()
    assert settings.tier == "full"
    assert settings.chat.model == "from-env"


def test_ollama_style_base_url_needs_no_key():
    settings = Settings.load(
        overrides={
            "provider": "openai",
            "base_url": "http://localhost:11434/v1",
            "model": "llama3.2",
        }
    )
    assert settings.chat.api_key() is None
    assert settings.chat.base_url == "http://localhost:11434/v1"


def test_rejects_unknown_tier(isolated_env):
    (isolated_env / "reelreader.toml").write_text('tier = "cinema"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="tier must be one of"):
        Settings.load()


def test_rejects_anthropic_embeddings(isolated_env):
    (isolated_env / "reelreader.toml").write_text(
        '[embed]\nprovider = "anthropic"\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="does not provide embeddings"):
        Settings.load()


def test_as_dict_never_includes_the_key(isolated_env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    payload = Settings.load().as_dict()
    assert payload["chat"]["api_key_present"] is True
    assert "sk-secret-value" not in str(payload)
