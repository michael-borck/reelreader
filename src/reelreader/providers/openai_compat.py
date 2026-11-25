"""Any OpenAI-compatible /chat/completions endpoint.

Ollama, LM Studio, and vLLM all speak this protocol. A key is optional:
when it is absent the Authorization header is omitted, which is what local
runtimes expect.
"""

from __future__ import annotations

import httpx

from reelreader.providers.base import ChatRequest, ProviderError

DEFAULT_TIMEOUT = 60.0


def _normalise_base(base_url: str) -> str:
    return base_url.rstrip("/")


class OpenAICompatibleProvider:
    requires_key = False

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        label: str = "openai-compatible",
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = _normalise_base(base_url)
        self.model = model
        self.name = label
        self._api_key = api_key
        self._timeout = timeout
        self._client = client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _request(self, method: str, path: str, payload: dict | None = None) -> httpx.Response:
        url = f"{self.base_url}{path}"
        if self._client is not None:
            return self._client.request(method, url, json=payload, headers=self._headers())
        with httpx.Client(timeout=self._timeout) as client:
            return client.request(method, url, json=payload, headers=self._headers())

    def _check(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        detail = " ".join(response.text.split())[:300]
        raise ProviderError(
            f"{self.name} returned HTTP {response.status_code}: {detail or 'no body'}"
        )

    def complete(self, request: ChatRequest) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        response = self._request("POST", "/chat/completions", payload)
        self._check(response)
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(f"{self.name} returned an unexpected response: {exc}") from exc
        if not content or not str(content).strip():
            raise ProviderError(f"{self.name} returned an empty completion")
        return str(content)

    def list_models(self) -> list[str]:
        response = self._request("GET", "/models")
        self._check(response)
        try:
            data = response.json().get("data", [])
        except (TypeError, ValueError) as exc:
            raise ProviderError(f"{self.name} returned an unexpected model list: {exc}") from exc
        return sorted({str(item["id"]) for item in data if item.get("id")})
