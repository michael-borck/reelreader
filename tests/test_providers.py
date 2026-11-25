from __future__ import annotations

import httpx
import pytest

from reelreader.providers import ChatRequest
from reelreader.providers.anthropic import AnthropicProvider
from reelreader.providers.base import MissingKeyError, ProviderError
from reelreader.providers.embedders import OpenAICompatibleEmbedder
from reelreader.providers.openai_compat import OpenAICompatibleProvider

REQUEST = ChatRequest(system="be brief", user="what is 2+2?")


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_omits_bearer_when_keyless():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "4"}}]})

    provider = OpenAICompatibleProvider(
        base_url="http://localhost:11434/v1/",
        model="llama3.2",
        client=mock_client(handler),
    )
    assert provider.complete(REQUEST) == "4"
    assert "authorization" not in seen[0].headers
    assert str(seen[0].url) == "http://localhost:11434/v1/chat/completions"
    assert seen[0].read().decode().find('"model":"llama3.2"') != -1


def test_openai_sends_bearer_when_keyed():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = OpenAICompatibleProvider(
        base_url="https://api.openai.com/v1",
        model="gpt-4.1-mini",
        api_key="sk-test",
        client=mock_client(handler),
    )
    provider.complete(REQUEST)
    assert seen[0].headers["authorization"] == "Bearer sk-test"


def test_openai_surfaces_http_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "nope"})

    provider = OpenAICompatibleProvider(
        base_url="https://api.openai.com/v1",
        model="m",
        client=mock_client(handler),
    )
    with pytest.raises(ProviderError, match="HTTP 401"):
        provider.complete(REQUEST)


def test_openai_rejects_empty_completion():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    provider = OpenAICompatibleProvider(
        base_url="https://api.openai.com/v1",
        model="m",
        client=mock_client(handler),
    )
    with pytest.raises(ProviderError, match="empty completion"):
        provider.complete(REQUEST)


def test_anthropic_headers_and_parse():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "four"}]},
        )

    provider = AnthropicProvider(
        base_url="https://api.anthropic.com",
        model="claude-sonnet-4-5",
        api_key="sk-ant-test",
        client=mock_client(handler),
    )
    assert provider.complete(REQUEST) == "four"
    request = seen[0]
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "sk-ant-test"
    assert request.headers["anthropic-version"] == "2023-06-01"
    body = request.read().decode()
    assert '"system":"be brief"' in body


def test_anthropic_requires_key():
    provider = AnthropicProvider(
        base_url="https://api.anthropic.com",
        model="claude-sonnet-4-5",
        api_key=None,
    )
    with pytest.raises(MissingKeyError, match="ANTHROPIC_API_KEY"):
        provider.complete(REQUEST)


def test_model_listing_is_sorted_and_deduped():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"id": "b"}, {"id": "a"}, {"id": "b"}]},
        )

    provider = OpenAICompatibleProvider(
        base_url="https://api.openai.com/v1",
        model="m",
        client=mock_client(handler),
    )
    assert provider.list_models() == ["a", "b"]


def test_openai_embedder_orders_by_index():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            },
        )

    embedder = OpenAICompatibleEmbedder(
        base_url="https://api.openai.com/v1",
        model="text-embedding-3-small",
        dim=2,
        client=mock_client(handler),
    )
    assert embedder.embed_texts(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
