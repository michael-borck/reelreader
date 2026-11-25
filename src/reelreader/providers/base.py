"""Shared provider types. Transport-specific code lives in the siblings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class ProviderError(RuntimeError):
    """Any provider failure with enough context to show the user."""


class MissingKeyError(ProviderError):
    """The provider needs an API key and none was configured."""


@dataclass(frozen=True, slots=True)
class ChatRequest:
    system: str
    user: str
    max_tokens: int = 1024
    temperature: float = 0.2


@runtime_checkable
class ChatProvider(Protocol):
    name: str
    model: str
    requires_key: bool

    def complete(self, request: ChatRequest) -> str: ...

    def list_models(self) -> list[str]: ...
