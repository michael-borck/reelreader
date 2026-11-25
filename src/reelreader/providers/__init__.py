"""Chat and embedding providers.

Two chat options, on purpose: any OpenAI-compatible endpoint (which includes
Ollama, LM Studio, vLLM, OpenRouter, Groq, and OpenAI itself) and Anthropic.
Embeddings are a separate choice because Anthropic does not provide them.
"""

from __future__ import annotations

from reelreader.config import Settings
from reelreader.providers.anthropic import AnthropicProvider
from reelreader.providers.base import (
    ChatProvider,
    ChatRequest,
    MissingKeyError,
    ProviderError,
)
from reelreader.providers.embedders import Embedder, build_embedder
from reelreader.providers.openai_compat import OpenAICompatibleProvider

__all__ = [
    "AnthropicProvider",
    "ChatProvider",
    "ChatRequest",
    "Embedder",
    "MissingKeyError",
    "OpenAICompatibleProvider",
    "ProviderError",
    "build_chat_provider",
    "build_embedder",
]


def build_chat_provider(settings: Settings) -> ChatProvider:
    chat = settings.chat
    if chat.provider == "anthropic":
        return AnthropicProvider(
            base_url=chat.base_url,
            model=chat.model,
            api_key=chat.api_key(),
        )
    if chat.provider == "openai":
        return OpenAICompatibleProvider(
            base_url=chat.base_url,
            model=chat.model,
            api_key=chat.api_key(),
            label="openai",
        )
    raise ProviderError(f"unknown chat provider: {chat.provider!r}")
