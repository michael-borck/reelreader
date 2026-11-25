from __future__ import annotations

import pytest

from reelreader.config import Settings
from reelreader.providers.base import ProviderError
from reelreader.rag import RagError, ask, build_prompt, parse_citations
from reelreader.search import Hit


class StubEmbedder:
    name = "stub"
    model = "stub"
    dim = 2

    def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]


class StubProvider:
    name = "stub"
    model = "stub-model"
    requires_key = False

    def __init__(self, reply: str = "", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.reply

    def list_models(self):
        return []


def hit(index: int, text: str, *, start: float = 0.0, dense: float = 0.9) -> Hit:
    return Hit(
        segment_id=f"s{index}",
        video_id="v",
        video_title="Lecture",
        start=start,
        end=start + 20,
        text=text,
        score=dense,
        dense=dense,
        bm25=1.0,
    )


HITS = [
    hit(
        1,
        "A p value measures how likely the data are under the null hypothesis.",
        start=0.0,
        dense=0.9,
    ),
    hit(2, "If the p value is below alpha we reject the null hypothesis.", start=20.0, dense=0.8),
    hit(3, "We take a coffee break.", start=40.0, dense=0.7),
]


def ask_with(monkeypatch, reply: str, hits=None, provider=None, **kwargs):
    monkeypatch.setattr(
        "reelreader.rag.search_library",
        lambda *args, **kw: HITS if hits is None else hits,
    )
    engine = provider or StubProvider(reply=reply)
    answer = ask(
        library=None,
        settings=Settings.load(),
        question="What is a p value?",
        provider=engine,
        embedder=StubEmbedder(),
        **kwargs,
    )
    return answer, engine


def test_prompt_contains_only_retrieved_text():
    prompt = build_prompt("What is a p value?", HITS)
    assert "Question: What is a p value?" in prompt
    assert "[1] 0:00-0:20" in prompt
    assert HITS[0].text in prompt
    assert HITS[2].text in prompt


def test_citations_map_back_to_segments(monkeypatch):
    answer, engine = ask_with(monkeypatch, "A p-value is the probability of the data [1].")
    assert answer.abstained is False
    assert answer.citations[0].segment_id == "s1"
    assert answer.citations[0].label == "0:00"
    assert len(engine.requests) == 1
    assert HITS[0].text in engine.requests[0].user
    assert "capital of France" not in engine.requests[0].user


def test_citations_are_deduped_and_ordered(monkeypatch):
    answer, _ = ask_with(
        monkeypatch,
        "First [2], then [1], and again [2].",
    )
    assert [citation.segment_id for citation in answer.citations] == ["s2", "s1"]


def test_invalid_citation_abstains(monkeypatch):
    answer, _ = ask_with(monkeypatch, "The answer is [9].")
    assert answer.abstained is True
    assert answer.reason == "no_citations"


def test_model_abstention(monkeypatch):
    answer, _ = ask_with(monkeypatch, "NOT_COVERED")
    assert answer.abstained is True
    assert answer.reason == "model_abstained"
    assert answer.citations[0].segment_id == "s1"
    assert "0:00" in answer.text


def test_low_score_skips_the_model(monkeypatch):
    weak = [hit(1, "unrelated", dense=0.1)]
    answer, engine = ask_with(monkeypatch, "should not be called", hits=weak)
    assert answer.abstained is True
    assert answer.reason == "low_score"
    assert engine.requests == []


def test_no_hits_abstains(monkeypatch):
    answer, engine = ask_with(monkeypatch, "unused", hits=[])
    assert answer.abstained is True
    assert answer.reason == "no_hits"
    assert answer.citations == ()
    assert engine.requests == []


def test_empty_question_is_rejected():
    with pytest.raises(RagError, match="Ask a question"):
        ask(
            library=None,
            settings=Settings.load(),
            question="  ",
            provider=StubProvider(),
            embedder=StubEmbedder(),
        )


def test_provider_errors_propagate(monkeypatch):
    provider = StubProvider(error=ProviderError("boom"))
    with pytest.raises(ProviderError, match="boom"):
        ask_with(monkeypatch, "", provider=provider)


def test_parse_citations_ignores_out_of_range():
    citations = parse_citations("[0] [1] [99]", HITS)
    assert [citation.segment_id for citation in citations] == ["s1"]
