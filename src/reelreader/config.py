"""Resolved settings for one Reelreader run.

Precedence, highest first: environment variables, the nearest
``reelreader.toml``, built-in defaults. API keys are never read from TOML.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

CONFIG_NAME = "reelreader.toml"
ENV_PREFIX = "REELREADER_"

TIERS = ("text", "visual", "full")
CHAT_PROVIDERS = ("openai", "anthropic")
EMBED_PROVIDERS = ("local", "openai")


class ConfigError(RuntimeError):
    """The configuration file or an override is invalid."""


def default_library() -> Path:
    return Path.home() / "Reelreader"


def config_candidates(start: Path | None = None) -> list[Path]:
    here = (start or Path.cwd()).resolve()
    candidates = [here / CONFIG_NAME]
    candidates.extend(parent / CONFIG_NAME for parent in here.parents)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        candidates.append(Path(xdg) / "reelreader" / CONFIG_NAME)
    candidates.append(Path.home() / ".config" / "reelreader" / CONFIG_NAME)
    return candidates


def find_config(start: Path | None = None) -> Path | None:
    for path in config_candidates(start):
        if path.is_file():
            return path
    return None


@dataclass(frozen=True, slots=True)
class ChatSettings:
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 1024

    def api_key(self) -> str | None:
        value = os.environ.get(self.api_key_env, "").strip()
        return value or None


@dataclass(frozen=True, slots=True)
class EmbedSettings:
    provider: str = "local"
    base_url: str = "https://api.openai.com/v1"
    model: str = "BAAI/bge-small-en-v1.5"
    api_key_env: str = "OPENAI_API_KEY"
    dim: int = 384

    def api_key(self) -> str | None:
        value = os.environ.get(self.api_key_env, "").strip()
        return value or None


@dataclass(frozen=True, slots=True)
class Settings:
    library: Path = field(default_factory=default_library)
    tier: str = "text"
    chat: ChatSettings = field(default_factory=ChatSettings)
    embed: EmbedSettings = field(default_factory=EmbedSettings)
    config_path: Path | None = None

    @classmethod
    def load(cls, start: Path | None = None, overrides: dict[str, Any] | None = None) -> Settings:
        path = find_config(start)
        raw: dict[str, Any] = {}
        if path is not None:
            try:
                raw = tomllib.loads(path.read_text(encoding="utf-8"))
            except tomllib.TOMLDecodeError as exc:
                raise ConfigError(f"{path}: {exc}") from exc
        settings = cls._from_mapping(raw, path)
        settings = settings._with_env()
        if overrides:
            settings = settings._with_overrides(overrides)
        settings.validate()
        return settings

    @classmethod
    def _from_mapping(cls, raw: dict[str, Any], path: Path | None) -> Settings:
        chat_raw = _section(raw, "chat")
        embed_raw = _section(raw, "embed")
        chat_defaults = ChatSettings()
        embed_defaults = EmbedSettings()
        return cls(
            library=_path(raw.get("library")) or default_library(),
            tier=str(raw.get("tier", "text")),
            chat=ChatSettings(
                provider=str(chat_raw.get("provider", chat_defaults.provider)),
                base_url=str(chat_raw.get("base_url", chat_defaults.base_url)),
                model=str(chat_raw.get("model", chat_defaults.model)),
                api_key_env=str(chat_raw.get("api_key_env", chat_defaults.api_key_env)),
                temperature=float(chat_raw.get("temperature", chat_defaults.temperature)),
                max_tokens=int(chat_raw.get("max_tokens", chat_defaults.max_tokens)),
            ),
            embed=EmbedSettings(
                provider=str(embed_raw.get("provider", embed_defaults.provider)),
                base_url=str(embed_raw.get("base_url", embed_defaults.base_url)),
                model=str(embed_raw.get("model", embed_defaults.model)),
                api_key_env=str(embed_raw.get("api_key_env", embed_defaults.api_key_env)),
                dim=int(embed_raw.get("dim", embed_defaults.dim)),
            ),
            config_path=path,
        )

    def _with_env(self) -> Settings:
        chat = self.chat
        embed = self.embed
        chat_updates = {
            "provider": _env("CHAT_PROVIDER"),
            "base_url": _env("CHAT_BASE_URL"),
            "model": _env("CHAT_MODEL"),
            "api_key_env": _env("CHAT_API_KEY_ENV"),
            "temperature": _env_float("CHAT_TEMPERATURE"),
            "max_tokens": _env_int("CHAT_MAX_TOKENS"),
        }
        embed_updates = {
            "provider": _env("EMBED_PROVIDER"),
            "base_url": _env("EMBED_BASE_URL"),
            "model": _env("EMBED_MODEL"),
            "api_key_env": _env("EMBED_API_KEY_ENV"),
            "dim": _env_int("EMBED_DIM"),
        }
        chat = replace(chat, **{k: v for k, v in chat_updates.items() if v is not None})
        embed = replace(embed, **{k: v for k, v in embed_updates.items() if v is not None})
        return replace(
            self,
            library=_path(_env("LIBRARY")) or self.library,
            tier=_env("TIER") or self.tier,
            chat=chat,
            embed=embed,
        )

    def _with_overrides(self, overrides: dict[str, Any]) -> Settings:
        chat_fields = {item.name for item in fields(ChatSettings)}
        embed_fields = {item.name for item in fields(EmbedSettings)}
        chat = replace(
            self.chat,
            **{k: v for k, v in overrides.items() if k in chat_fields},
        )
        embed = replace(
            self.embed,
            **{k: v for k, v in overrides.items() if k in embed_fields},
        )
        top = {k: v for k, v in overrides.items() if k in {"library", "tier"} and v is not None}
        if "library" in top:
            top["library"] = Path(top["library"]).expanduser()
        return replace(self, chat=chat, embed=embed, **top)

    def validate(self) -> None:
        if self.tier not in TIERS:
            raise ConfigError(f"tier must be one of {', '.join(TIERS)}, got {self.tier!r}")
        if self.chat.provider not in CHAT_PROVIDERS:
            raise ConfigError(
                f"chat.provider must be one of {', '.join(CHAT_PROVIDERS)}, "
                f"got {self.chat.provider!r}"
            )
        if self.embed.provider not in EMBED_PROVIDERS:
            raise ConfigError(
                "embed.provider must be one of "
                f"{', '.join(EMBED_PROVIDERS)}, got {self.embed.provider!r}. "
                "Use an OpenAI-compatible endpoint or 'local'; "
                "Anthropic does not provide embeddings."
            )
        if self.embed.dim <= 0:
            raise ConfigError("embed.dim must be a positive integer")

    def as_dict(self) -> dict[str, Any]:
        return {
            "library": str(self.library),
            "tier": self.tier,
            "config_path": str(self.config_path) if self.config_path else None,
            "chat": {
                "provider": self.chat.provider,
                "base_url": self.chat.base_url,
                "model": self.chat.model,
                "api_key_env": self.chat.api_key_env,
                "api_key_present": self.chat.api_key() is not None,
                "temperature": self.chat.temperature,
                "max_tokens": self.chat.max_tokens,
            },
            "embed": {
                "provider": self.embed.provider,
                "base_url": self.embed.base_url,
                "model": self.embed.model,
                "api_key_env": self.embed.api_key_env,
                "api_key_present": self.embed.api_key() is not None,
                "dim": self.embed.dim,
            },
        }


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{key}] must be a table")
    return value


def _path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()


def _env(name: str) -> str | None:
    value = os.environ.get(ENV_PREFIX + name, "").strip()
    return value or None


def _env_int(name: str) -> int | None:
    value = _env(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{ENV_PREFIX}{name} must be an integer") from exc


def _env_float(name: str) -> float | None:
    value = _env(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"{ENV_PREFIX}{name} must be a number") from exc
