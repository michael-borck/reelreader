# Context

Vocabulary and invariants for anyone (human or agent) changing Reelreader.
Read this before the code; read `docs/SPEC.md` before changing behaviour.

## Vocabulary

- **Library** — the root folder holding videos, the database, and the index.
  Default `~/Reelreader`. A library is portable: copy the folder, keep the data.
- **Video** — one source file and everything derived from it, under
  `videos/<id>/`. Identified by a stable id, deduplicated by `source_sha256`.
- **Segment** — a 20–40 s slice that ends on a sentence or a scene change. The
  unit of retrieval and citation. Has a start, end, text, and artifacts.
- **Moment** — the user-facing word for a segment. Do not say "chunk" in UI.
- **View** — one way a segment is embedded: `text`, `frame`, `video`, `audio`,
  `fused`. One shared vector space, several inputs.
- **Tier** — how many views are indexed: `text`, `visual` (text+frame), `full`
  (all). A hardware and cost decision, surfaced by `doctor`.
- **Manifest** — `index/manifest.json`. Records model, revision, dimension, and
  normalisation per view. A mismatch is an error, never a silent mix.
- **Citation** — `[#<segment_id> @ mm:ss]`. Must map to a retrieved segment.
- **Abstention** — the honest "not covered" answer with the nearest moment. A
  feature, not a failure.
- **Effective key** — the API key resolved from env or keychain for one request.
  Lives in the provider object only; never serialised.

## Invariants

1. The index is derived data. Deleting `index/` loses time, not content.
2. The transcript and segment artifacts are the source of truth for retrieval.
3. A model or dimension change invalidates the index. Refuse to query a stale
   manifest; offer a rebuild.
4. Chat and embeddings are configured separately. Anthropic is chat-only.
5. The OpenAI-compatible provider omits `Authorization` when no key is set.
   Ollama depends on this.
6. Keys never enter TOML, logs, the renderer, or an HTTP response.
7. Every answer either cites a retrieved segment or abstains.
8. User data lives in the library folder, in open formats, inspectable without
   the app.
9. The desktop shell never implements retrieval; it calls the engine.
10. No telemetry. Outbound calls go only to the configured provider hosts.
11. The model cites segment numbers; only the server maps them to ids and
    timestamps. An answer with no valid citation becomes an abstention.

## Module map

| Path | Responsibility |
|---|---|
| `src/reelreader/config.py` | settings, precedence, validation |
| `src/reelreader/cli.py` | command surface |
| `src/reelreader/ids.py` | sortable ULID identifiers |
| `src/reelreader/library.py` | SQLite schema, videos, jobs, migrations |
| `src/reelreader/media.py` | ffprobe inspection |
| `src/reelreader/ingest.py` | hash, dedupe, folder and URL ingest |
| `src/reelreader/transcribe.py` | words, sentences, faster-whisper adapter |
| `src/reelreader/segment.py` | boundaries, clips, wavs, keyframes |
| `src/reelreader/pipeline.py` | resumable transcode → transcribe → segment |
| `src/reelreader/embed.py` | text embeddings, index matrices, manifest |
| `src/reelreader/search.py` | dense, BM25, and RRF hybrid retrieval |
| `src/reelreader/rag.py` | grounded prompt, citation mapping, abstention |
| `src/reelreader/providers/base.py` | shared provider types and errors |
| `src/reelreader/providers/openai_compat.py` | OpenAI-compatible chat |
| `src/reelreader/providers/anthropic.py` | Anthropic messages |
| `src/reelreader/providers/embedders.py` | local and OpenAI-compatible embeddings |
| `docs/SPEC.md` | the contract |
| `docs/UX.md` | the loop and states |
| `docs/STORIES.md` | backlog and definition of done |

Planned modules, named here so they are not invented twice: `serve.py`, and the
multimodal view embedders in M3–M4.

## Conventions

- Python 3.11+, `ruff` for lint and format, `pytest` for tests, `basedpyright`
  standard mode.
- Frozen dataclasses for settings and value types.
- Providers take an injectable `httpx.Client` so tests use `MockTransport` and
  no network.
- Errors carry the thing and the fix: which provider, which status, which env
  var.
- No comments that restate the code; docstrings explain why a module exists.
- Pin `av<19`: faster-whisper 1.2.x passes `metadata_errors` to `av.open`, which
  PyAV 19 removed. A newer faster-whisper may lift the pin; verify before moving
  it.
- Artifact clips are video-only and encoded `ultrafast`; per-moment audio lives
  in the sibling `.wav`, so a clip does not pay for an audio encode.
