"""Embedding backends for the index.

``local`` runs on the machine and needs no key, which is the default so that
indexing a lecture is free and works offline. ``openai`` talks to any
OpenAI-compatible ``/embeddings`` endpoint.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx

from reelreader.config import Settings
from reelreader.providers.base import ProviderError

DEFAULT_TIMEOUT = 120.0


@runtime_checkable
class Embedder(Protocol):
    name: str
    model: str
    dim: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbedder:
    name = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dim: int,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        self._api_key = api_key
        self._timeout = timeout
        self._client = client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.model, "input": texts}
        url = f"{self.base_url}/embeddings"
        if self._client is not None:
            response = self._client.post(url, json=payload, headers=self._headers())
        else:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, json=payload, headers=self._headers())
        if not response.is_success:
            detail = " ".join(response.text.split())[:300]
            raise ProviderError(
                f"{self.name} embeddings returned HTTP {response.status_code}: "
                f"{detail or 'no body'}"
            )
        try:
            rows = sorted(response.json()["data"], key=lambda row: row["index"])
            return [[float(value) for value in row["embedding"]] for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError(
                f"{self.name} returned an unexpected embedding payload: {exc}"
            ) from exc


class LocalEmbedder:
    name = "local"

    def __init__(self, *, model: str, dim: int) -> None:
        import importlib

        try:
            module = importlib.import_module("fastembed")
        except ImportError as exc:
            raise ProviderError(
                "local embeddings need the optional dependency: pip install 'reelreader[local]'"
            ) from exc
        self.model = model
        self.dim = dim
        self._model = module.TextEmbedding(model_name=model)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [[float(value) for value in vector] for vector in self._model.embed(texts)]


def build_embedder(settings: Settings) -> Embedder:
    embed = settings.embed
    if embed.provider == "local":
        return LocalEmbedder(model=embed.model, dim=embed.dim)
    if embed.provider == "openai":
        return OpenAICompatibleEmbedder(
            base_url=embed.base_url,
            model=embed.model,
            dim=embed.dim,
            api_key=embed.api_key(),
        )
    raise ProviderError(
        f"unknown embedding provider: {embed.provider!r}. "
        "Anthropic does not provide embeddings; use 'local' or an "
        "OpenAI-compatible endpoint."
    )
