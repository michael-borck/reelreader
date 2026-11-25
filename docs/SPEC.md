# Reelreader specification

Status: draft, v0.0.1. This document is the contract; code follows it.

## 1. Summary

Reelreader indexes a video as the parts a person would point at — transcript
lines, keyframes, audio, short clips — and answers questions with citations that
seek the player to the exact moment.

It ships as one Python engine with three surfaces:

| Surface | Distribution | Audience |
|---|---|---|
| `reelreader` CLI | PyPI (`uv tool`, `pipx`, `pip`) | scripts, batch, power users |
| local HTTP API | `reelreader serve` | the desktop shell, integrations |
| desktop app | signed Tauri build | students and lecturers |

The engine is written once. The desktop shell is a client, not a second
implementation.

## 2. Problem

A recorded lecture is a 60–120 minute file. A student who remembers "she said
something about the assignment weighting" has no way to get from that memory to
the minute. Existing tools fail in one of three ways:

1. **Search returns titles.** YouTube search, a file name, a chapter heading.
2. **Chatbots hallucinate.** A general model has never seen the video and will
   invent a plausible answer.
3. **Transcripts are walls of text.** Even a good transcript does not let you
   ask a question and land on the answer.

The missing product is retrieval plus a seekable citation.

## 3. Users

- **Student** — downloads or receives recorded lectures, revises, wants the
  moment and a study guide.
- **Lecturer** — wants to know what was actually said, build clips, check
  coverage, export a transcript.
- **Researcher / analyst** — a folder of talks or interviews, cross-video
  questions.
- **Developer** — wants a CLI and an API, not a GUI.

Primary persona for v1: the student revising alone on a laptop.

## 4. Goals

- Ask a question about one video or the whole library and get an answer grounded
  in retrieved segments, with timestamps that seek.
- Work with no account, no key, and no network when a local model is available.
- Index the multimodal signal: words, keyframes, audio, and short clips.
- Run on a normal laptop; degrade to a cheaper tier rather than fail.
- One engine, three surfaces, one data format the user can inspect and copy.

## 5. Non-goals (v1)

- Live capture or screen recording.
- Video editing, clipping export beyond a single moment.
- Multi-user collaboration, accounts, roles, or a hosted service.
- Automatic lecture capture or LMS integration.
- Fine-tuning or training models.
- Real-time meeting assistance.

## 6. Principles

1. **Local-first.** Video, transcript, index, and chat history stay on disk. No
   telemetry. The only outbound calls go to the embedding and chat endpoints the
   user configured.
2. **Seekable citations or no answer.** Every claim maps to a segment with a
   start and end time, and the UI can seek to it.
3. **Two chat providers, on purpose.** OpenAI-compatible and Anthropic. Local
   runtimes speak OpenAI-compatible, so "local" is not a third code path.
4. **Embeddings are independent of chat.** Anthropic has no embeddings API, so
   the index uses a local model by default or an OpenAI-compatible embedder.
5. **Honest degradation.** If the machine or the data cannot support a tier, say
   so before indexing, not after.
6. **The user owns the folder.** A library is a directory. Copy it, back it up,
   delete it without the app.

## 7. Pipeline

```
add ──▶ probe ──▶ transcode ──▶ transcribe ──▶ segment ──▶ embed ──▶ index
                      │              │            │           │
                   proxy.mp4    words+times   segments/   *.npy + manifest
                                              artifacts
```

Each stage is idempotent, resumable, and records its output in the library
database. An interrupted index resumes at the first missing artifact, not from
zero.

### 7.1 Probe

Read duration, streams, frame rate, resolution, and audio presence with ffprobe.
Reject or warn on: no audio track (transcript tier impossible), DRM, unsupported
container. Record `source_sha256` so a re-add is a no-op.

### 7.2 Transcode

Produce a proxy for playback and a normalised audio track for ASR:

- `proxy.mp4` — 480p, H.264, AAC, faststart. The player uses this; the source is
  never modified.
- `audio.wav` — 16 kHz mono PCM for ASR and the `audio` view.

### 7.3 Transcribe

Default engine: `faster-whisper` with word timestamps. Optional: WhisperX for
word alignment and diarisation, or a provider ASR if one is configured.

Output `transcript.json`:

```json
{
  "language": "en",
  "engine": "faster-whisper:small",
  "words": [{"start": 1.02, "end": 1.31, "text": "Today"}],
  "sentences": [{"start": 0.98, "end": 4.40, "text": "Today we cover p-values."}]
}
```

Word times are the unit of citation precision. Sentences exist so segments can
end on a sentence boundary.

### 7.4 Segment

Target 20–40 seconds, never crossing a strong boundary:

- a sentence end,
- a detected scene or slide change (`scenedetect`),
- a silence longer than 1.2 s.

Each segment gets an id, start/end, text, word span, and artifacts:

```
segments/0007.mp4    clip for the video view
segments/0007.wav    16 kHz mono audio for the audio view
segments/0007.jpg    keyframe at the midpoint (visual tier)
segments/0007.json   text, times, keyframe offsets
```

Boundaries are stored, not recomputed. Re-segmenting is an explicit command that
invalidates the index.

### 7.5 Embed

For each configured view, embed every segment and write one matrix per view plus
a manifest. See §9.

### 7.6 Index

Build the retrieval structures: cosine over each view, BM25 over segment text,
and the fusion weights. The index is derived data; it can always be rebuilt from
the transcript and artifacts.

## 8. Storage layout

```
~/Reelreader/                      library root (configurable)
├── library.db                     SQLite: videos, segments, jobs, chats
├── videos/<video_id>/
│   ├── source.mp4                 the original, untouched
│   ├── proxy.mp4
│   ├── audio.wav
│   ├── transcript.json
│   ├── segments.json
│   ├── thumb.jpg
│   └── segments/0001.{mp4,wav,jpg,json}
└── index/
    ├── manifest.json
    └── <video_id>/<view>.npy
```

`library.db` tables:

| Table | Columns (essential) |
|---|---|
| `videos` | id, path, sha256, title, duration, added_at, status |
| `segments` | id, video_id, start, end, text, keyframe_at |
| `jobs` | id, video_id, stage, state, error, updated_at |
| `chats` | id, video_id, question, answer, citations, created_at |

The index is never the source of truth. Deleting `index/` costs time, not data.

## 9. Embedding views

Following the multimodal design that samwit's video-RAG experiment validated:
one shared space, several ways to look at a segment.

| View | Input | Answers |
|---|---|---|
| `text` | `title + transcript` | spoken content |
| `frame` | one keyframe image | "the slide with the formula" |
| `video` | frames at 1 fps, no audio | what changes on screen |
| `audio` | raw 16 kHz audio | prosody and non-speech |
| `fused` | transcript + video + audio tokens in one input | cross-modal questions |

Retrieval combines them with reciprocal rank fusion (RRF, k=60) as `hybrid`, and
supports per-view search for diagnosis.

Model: **EmbeddingGemma 2** (768d) for the multimodal views. Text tier may use a
smaller text model (`bge-small-en-v1.5`, 384d) because it is fast on CPU.

Rules learned the hard way, and enforced here:

- Run the embedder in **bf16 or fp32**. fp16 silently produces NaN or degraded
  vectors.
- After Matryoshka (MRL) truncation to 512/256/128, **re-normalise**. Store the
  full 768d vector and truncate at query time.
- Record the model id, revision, dimension, and normalisation in the manifest. A
  model change invalidates the index; refuse to mix.

## 10. Retrieval tiers

Full multimodal is the destination, but a laptop is the constraint. The tier is a
single setting and is reported by `reelreader doctor`.

| Tier | Views | Index cost (1 h video) | Hardware |
|---|---|---|---|
| `text` | text | ~1 min, <200 MB | any CPU |
| `visual` | text, frame | ~5–15 min, ~1 GB | modern CPU or integrated GPU |
| `full` | text, frame, video, audio, fused | ~20–60 min, several GB | discrete GPU, 8 GB+ VRAM |

The app estimates cost from duration and the chosen tier before starting, and
lets the user queue overnight.

## 11. Index manifest

```json
{
  "format": 1,
  "video_id": "01J...",
  "source_sha256": "…",
  "tier": "full",
  "views": {
    "text":  {"model": "google/embeddinggemma-2", "rev": "…", "dim": 768, "normalised": true},
    "frame": {"model": "google/embeddinggemma-2", "rev": "…", "dim": 768, "normalised": true}
  },
  "segment_count": 214,
  "created_at": "2026-11-25T09:00:00Z"
}
```

Any query checks the manifest first. A mismatch between the configured embedder
and the manifest is an error with a one-command fix (`reelreader index --rebuild`).

## 12. Retrieval and answering

1. Embed the question in every view the tier provides.
2. Retrieve top 8 per view by cosine; top 8 by BM25 for text.
3. Fuse ranks with RRF; keep the top 8 segments after de-duplicating adjacent
   segments (merge neighbours into one window of ±1 segment).
4. Assemble context: segment text, timestamps, and — for visual/full — the
   keyframe and a short clip reference.
5. Call the chat model with a strict contract (§12.1).
6. Parse citations of the form `[#<segment_id> @ mm:ss]`; reject or repair
   citations that do not map to a retrieved segment.
7. Return answer, citations, and the retrieval trace.

### 12.1 Grounding contract

The system prompt says, in effect:

- Answer only from the provided segments.
- Cite every factual sentence with a segment id and its start time.
- If the segments do not contain the answer, say "This is not covered in the
  video" and name the nearest moment. Do not guess.
- Keep the answer under the requested word budget.

The server enforces the contract by validating citations after generation. A
response with no valid citation is downgraded to the abstention message rather
than shown as fact.

### 12.2 Abstention threshold

If the best fused score is below a calibrated threshold for the tier, skip the
chat call and return the abstention message with the nearest segment. The
threshold is a per-tier constant, tuned by the eval harness, not a magic number
in the UI.

## 13. Providers

### 13.1 Chat

| Provider id | Endpoint | Auth | Key needed |
|---|---|---|---|
| `openai` | any OpenAI-compatible `/chat/completions` | `Authorization: Bearer` if a key is set | no |
| `anthropic` | `/v1/messages` | `x-api-key` + `anthropic-version` | yes |

`openai` is the id for the protocol, not the company. Ollama, LM Studio, vLLM,
OpenRouter, Groq, and OpenAI all use it. Ollama ignores the key, so the header is
omitted when the key is empty — this is a tested invariant, not a comment.

Configuration:

```toml
[chat]
provider = "openai"
base_url = "http://localhost:11434/v1"
model = "llama3.2"
api_key_env = "OPENAI_API_KEY"
```

### 13.2 Embeddings

| Provider id | Endpoint | Notes |
|---|---|---|
| `local` | in-process | default; fastembed, or torch for multimodal |
| `openai` | `/embeddings` | any OpenAI-compatible service |

`anthropic` is rejected at config load with a message explaining that Anthropic
has no embeddings API. Chat model and embedding model are separate settings.

### 13.3 Key handling

- Read from `api_key_env` (default `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`).
- With the `keyring` extra, read from the OS keychain instead.
- Never read a key from TOML; never print one; `reelreader config` reports
  `api_key_present` only.
- Allow `http://` base URLs only for loopback hosts, so a key cannot be sent in
  clear text to a remote host by a typo.

## 14. CLI

```
reelreader init [--directory DIR] [--force]
reelreader doctor
reelreader config [--json]
reelreader models
reelreader add SOURCE [--title T] [--tier text|visual|full]
reelreader index [--view all|text|frame|video|audio|fused] [--rebuild]
reelreader list [--json]
reelreader search QUERY [--view VIEW] [--limit N]
reelreader ask QUESTION [--video ID] [--tier TIER] [--json] [--stream]
reelreader export ID --format md|srt|vtt|anki [--out PATH]
reelreader serve [--port 0] [--host 127.0.0.1]
reelreader eval [--dataset PATH] [--view VIEW]
```

Conventions: exit 0 success, 1 user/config error, 2 not implemented, 3 provider
failure. `--json` on every read command for scripting. Progress on stderr, data
on stdout.

## 15. HTTP API

Local only. Bound to `127.0.0.1`. The desktop shell starts it with a per-launch
bearer token and an ephemeral port.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | liveness, version, tier |
| `GET` | `/videos` | list with status |
| `POST` | `/videos` | add by path or URL |
| `GET` | `/videos/{id}` | metadata, transcript, segments |
| `POST` | `/videos/{id}/index` | start or resume a job |
| `GET` | `/jobs/{id}` | progress, stage, error |
| `GET` | `/events` | SSE progress stream |
| `POST` | `/ask` | question + optional video filter |
| `GET` | `/search` | keyword and semantic search |
| `POST` | `/export` | write a transcript or study guide |
| `GET` | `/media/{id}/{segment}` | serve a clip, frame, or wav |

Errors are `{"error": {"code": "...", "message": "..."}}` with a stable code
(`no_audio`, `index_stale`, `provider_auth`, `abstained`).

## 16. Desktop shell

Tauri v2. The shell owns the window, menus, file dialogs, and updates. The engine
runs as a sidecar:

1. Shell picks a free port and a random token.
2. Spawns `reelreader serve --port N --host 127.0.0.1` with the token in an env
   var, never on the command line.
3. Polls `/health` until ready, showing a splash with the stage.
4. Every request carries the bearer token; the token dies with the process.
5. On quit, terminate the sidecar and its children; on crash, offer restart.

The webview loads only bundled assets. No remote content, no arbitrary
navigation, a strict CSP. The sidecar is the only process that talks to a
provider.

## 17. Security and privacy

- Secrets: environment or OS keychain. Never in TOML, never in logs, never in the
  renderer.
- Trust boundary: validate every API request (types, ids, paths). Media paths are
  resolved inside the library root; no path traversal.
- Outbound: only the configured provider hosts. Block automatic redirects to
  other hosts. Refuse non-loopback `http://`.
- Rendered content: model output is Markdown only; sanitise links and HTML.
- Logs: request metadata only. Never log keys, full prompts, or transcript text
  by default; add `--debug` for local diagnosis.
- Telemetry: none.

## 18. Performance targets

Measured on a 1-hour 1080p lecture, Apple M-series, 16 GB:

| Operation | Target |
|---|---|
| probe + transcode | < 2 min |
| transcribe (small, CPU) | < 0.3× realtime |
| segment + keyframes | < 30 s |
| text index | < 60 s |
| query (retrieve + answer, 8 segments) | < 3 s first token |
| CLI cold start | < 400 ms |

The desktop must stay responsive while a job runs: jobs execute in the sidecar,
progress arrives over SSE, and the UI never blocks on a pipeline stage.

## 19. Evaluation

A retrieval claim without a number is marketing. The harness is part of v1.

- `data/eval/manual.jsonl`: real questions with `video_id`, `expected_segment`,
  and optional `expected_start`.
- Synthetic set: a chat model writes spoken and visual questions from sampled
  segments; flagged as synthetic in results.
- Metrics: recall@5 of the exact source segment, MRR, and answer groundedness
  (citations valid and sufficient, judged by a model and spot-checked by hand).
- Matrix: view × tier × MRL dimension, so the README can state what actually
  works.
- Gate: a change to segmentation, embedding, or fusion must not drop recall@5 by
  more than 2 points on the manual set.

Known bias to control: synthetic questions written from transcripts favour the
text view. Report spoken and visual queries separately.

## 20. Milestones

| Milestone | Ships |
|---|---|
| **M0 — skeleton** | config, providers, doctor, this spec (v0.0.1) |
| **M1 — text loop** | ingest, transcribe, segment, text index, seekable answer, CLI |
| **M2 — desktop** | Tauri shell, three-pane UI, library, player, transcript sync |
| **M3 — visual** | keyframes, frame view, filmstrip, clip citations |
| **M4 — full** | audio/video/fused views, tier estimates, eval matrix |
| **M5 — study** | exports (md, srt, vtt, anki), chapters, presenter window |

## 21. Open questions

1. **Playback engine.** Native `<video>` in the webview is simplest; subtitles,
   variable speed, and frame stepping are the risk. Decide in M2.
2. **Diarisation.** Needed for "who said it"; adds a heavy dependency. Behind an
   extra, off by default.
3. **Ollama model defaults.** Ship a recommendation per machine size, or let the
   user pick from `/models` every time?
4. **Library portability.** Absolute paths in the database break a moved folder;
   store relative paths and a root.
5. **Anthropic prompt caching.** Worth it for long transcripts; needs a
   chunking strategy that keeps citations stable.
6. **Licence of bundled models.** Confirm redistribution terms before bundling
   weights in the desktop installer; otherwise download on first use.
