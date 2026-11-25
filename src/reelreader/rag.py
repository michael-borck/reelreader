"""Grounded answering: retrieve moments, ask the model, validate citations.

The model sees only retrieved segments and cites them by number. The server
maps numbers back to segment ids, and an answer without a valid citation is
replaced by an abstention rather than shown as fact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from reelreader.config import Settings
from reelreader.providers import ChatRequest, build_chat_provider, build_embedder
from reelreader.providers.base import ChatProvider
from reelreader.providers.embedders import Embedder
from reelreader.search import Hit, format_time, search_library

MIN_DENSE_SCORE = 0.25
DEFAULT_LIMIT = 8
NOT_COVERED = "NOT_COVERED"

SYSTEM_PROMPT = """\
You answer questions about a recorded lecture using ONLY the numbered segments below.
Cite every factual sentence with the segment number in square brackets, for example [2].
Do not use outside knowledge.
If the segments do not contain the answer, reply with exactly NOT_COVERED and nothing else.
"""


class RagError(RuntimeError):
    """The question cannot be answered, with a reason the user can act on."""


@dataclass(frozen=True, slots=True)
class Citation:
    segment_id: str
    video_id: str
    video_title: str
    start: float
    end: float
    label: str
    snippet: str

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "video_id": self.video_id,
            "video_title": self.video_title,
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "snippet": self.snippet,
        }


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    abstained: bool
    citations: tuple[Citation, ...]
    segments: tuple[Hit, ...]
    model: str
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "answer": self.text,
            "abstained": self.abstained,
            "reason": self.reason,
            "model": self.model,
            "citations": [citation.to_dict() for citation in self.citations],
            "segments": [hit.to_dict() for hit in self.segments],
        }


def build_prompt(question: str, hits: list[Hit] | tuple[Hit, ...]) -> str:
    lines = [
        f"Question: {question.strip()}",
        "",
        "Segments:",
    ]
    for index, hit in enumerate(hits, start=1):
        lines.append(f"[{index}] {hit.label}-{format_time(hit.end)}")
        lines.append(hit.text.strip() or "(no speech in this moment)")
        lines.append("")
    return "\n".join(lines).strip()


def parse_citations(text: str, hits: list[Hit] | tuple[Hit, ...]) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[str] = set()
    for match in re.findall(r"\[(\d+)\]", text):
        position = int(match) - 1
        if not 0 <= position < len(hits):
            continue
        hit = hits[position]
        if hit.segment_id in seen:
            continue
        seen.add(hit.segment_id)
        citations.append(_citation(hit))
    return citations


def ask(
    library,
    settings: Settings,
    question: str,
    *,
    provider: ChatProvider | None = None,
    embedder: Embedder | None = None,
    limit: int = DEFAULT_LIMIT,
    video_id: str | None = None,
    mode: str = "hybrid",
) -> Answer:
    if not question.strip():
        raise RagError("Ask a question about the lecture.")
    engine = embedder or build_embedder(settings)
    hits = search_library(
        library,
        question,
        embedder=engine,
        limit=limit,
        mode=mode,
        video_id=video_id,
    )
    if not hits:
        return _abstain([], "no_hits", settings.chat.model)

    best = max(hits, key=lambda hit: hit.dense if hit.dense is not None else float("-inf"))
    if best.dense is not None and best.dense < MIN_DENSE_SCORE:
        return _abstain(hits, "low_score", settings.chat.model)

    chat = provider or build_chat_provider(settings)
    reply = chat.complete(
        ChatRequest(
            system=SYSTEM_PROMPT,
            user=build_prompt(question, hits),
            max_tokens=settings.chat.max_tokens,
            temperature=settings.chat.temperature,
        )
    )
    cleaned = reply.strip()
    if cleaned.upper().startswith(NOT_COVERED):
        return _abstain(hits, "model_abstained", settings.chat.model)

    citations = parse_citations(cleaned, hits)
    if not citations:
        return _abstain(hits, "no_citations", settings.chat.model)

    return Answer(
        text=cleaned,
        abstained=False,
        citations=tuple(citations),
        segments=tuple(hits),
        model=settings.chat.model,
    )


def _abstain(hits: list[Hit], reason: str, model: str) -> Answer:
    nearest = (
        max(hits, key=lambda hit: hit.dense if hit.dense is not None else float("-inf"))
        if hits
        else None
    )
    if nearest is None:
        text = "That is not covered in this video."
        citations: tuple[Citation, ...] = ()
    else:
        text = f"That is not covered in this video. The closest moment is {nearest.label}."
        citations = (_citation(nearest),)
    return Answer(
        text=text,
        abstained=True,
        citations=citations,
        segments=tuple(hits),
        model=model,
        reason=reason,
    )


def _citation(hit: Hit) -> Citation:
    snippet = " ".join(hit.text.split())[:160]
    return Citation(
        segment_id=hit.segment_id,
        video_id=hit.video_id,
        video_title=hit.video_title,
        start=hit.start,
        end=hit.end,
        label=hit.label,
        snippet=snippet,
    )
