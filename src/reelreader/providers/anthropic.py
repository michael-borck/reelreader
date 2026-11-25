"""Anthropic Messages API. Chat only; Anthropic has no embeddings endpoint."""

from __future__ import annotations

import httpx

from reelreader.providers.base import ChatRequest, MissingKeyError, ProviderError

DEFAULT_TIMEOUT = 60.0
ANTHROPIC_VERSION = "2023-06-01"


def _normalise_base(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


class AnthropicProvider:
    requires_key = True
    name = "anthropic"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = _normalise_base(base_url)
        self.model = model
        self._api_key = api_key
        self._timeout = timeout
        self._client = client

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise MissingKeyError(
                "Anthropic needs an API key. Set ANTHROPIC_API_KEY or change "
                "chat.api_key_env in reelreader.toml."
            )
        return {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }

    def _request(self, method: str, path: str, payload: dict | None = None) -> httpx.Response:
        url = f"{self.base_url}{path}"
        headers = self._headers()
        if self._client is not None:
            return self._client.request(method, url, json=payload, headers=headers)
        with httpx.Client(timeout=self._timeout) as client:
            return client.request(method, url, json=payload, headers=headers)

    def _check(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        detail = " ".join(response.text.split())[:300]
        raise ProviderError(
            f"anthropic returned HTTP {response.status_code}: {detail or 'no body'}"
        )

    def complete(self, request: ChatRequest) -> str:
        payload = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "system": request.system,
            "messages": [{"role": "user", "content": request.user}],
        }
        response = self._request("POST", "/messages", payload)
        self._check(response)
        try:
            blocks = response.json()["content"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError(f"anthropic returned an unexpected response: {exc}") from exc
        text = "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
        if not text:
            raise ProviderError("anthropic returned no text content")
        return text

    def list_models(self) -> list[str]:
        response = self._request("GET", "/models?limit=100")
        self._check(response)
        try:
            data = response.json().get("data", [])
        except (TypeError, ValueError) as exc:
            raise ProviderError(f"anthropic returned an unexpected model list: {exc}") from exc
        return sorted({str(item["id"]) for item in data if item.get("id")})
