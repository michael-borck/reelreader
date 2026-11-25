# Reelreader

Chat with recorded lectures and videos. Ask a question, get an answer, and click
the citation to jump to the moment it came from.

Reelreader is local-first: the video, transcript, keyframes, and index stay on
your machine. Your API key, if you use one, talks to the model you chose and
nothing else.

**Status: v0.0.1, pre-alpha.** The configuration, provider, and diagnostic
surface works. Indexing and retrieval land in the stories listed in
[docs/STORIES.md](docs/STORIES.md).

## Why

A recorded lecture is a two-hour file with a question somewhere inside it.
Search returns titles, not moments; a chatbot that never watched the video
invents an answer. Reelreader indexes the parts a person would point at
(transcript lines, keyframes, audio, clips) and answers with timestamps you can
seek to.

## Design in one screen

```
source video ─▶ transcode ─▶ transcribe ─▶ segment ─▶ embed ─▶ index
                                                             │
question ─▶ retrieve (text | frame | video | audio | fused) ─┤
                                                             ▼
                              grounded answer + [mm:ss] citations
```

- **One engine, three surfaces.** A Python core, a `reelreader` CLI on PyPI, a
  local HTTP API, and a Tauri desktop shell that is a thin client over the same
  API.
- **Two chat providers, on purpose.** Any OpenAI-compatible endpoint (OpenAI,
  Ollama, LM Studio, vLLM, OpenRouter, Groq) and Anthropic. Ollama needs no key,
  so the Authorization header is simply omitted.
- **Embeddings are a separate choice.** Anthropic has no embeddings API, so the
  index uses a local model by default (free, offline) or any OpenAI-compatible
  `/embeddings` endpoint.

## Install

```bash
uv tool install reelreader          # or: pipx install reelreader
```

Optional pieces are extras:

```bash
uv tool install 'reelreader[all]'   # local embeddings, ASR, scenes, serve, index
```

## Quickstart

```bash
reelreader init                     # write reelreader.toml, create ~/Reelreader
reelreader doctor                   # what is installed, which model, which tier
reelreader config                   # resolved settings, key redacted
reelreader models                   # models your provider offers

reelreader add lecture.mp4          # reference a file; --copy to bring it in
reelreader add ./week3 --copy       # a folder, copied into the library
reelreader list                     # videos, length, status, current job

reelreader index --model small      # transcode, transcribe, segment everything
reelreader index --video ID         # one video
reelreader index --rebuild          # ignore existing outputs
```

`index` resumes: a killed run picks up at the first missing artifact. Transcripts
land in `transcript.json`, moments in `segments.json` plus `segments/` clips,
wavs, and keyframes. Whisper models: `tiny`, `base`, `small`, `medium`,
`large-v3` (`small` by default).

Point at a local model with no key:

```toml
[chat]
provider = "openai"
base_url = "http://localhost:11434/v1"
model = "llama3.2"
```

Point at Anthropic:

```toml
[chat]
provider = "anthropic"
model = "claude-sonnet-4-5"
api_key_env = "ANTHROPIC_API_KEY"
```

## Retrieval tiers

Full multimodal is the destination; laptops are the constraint. The `tier`
setting says how much of the video to index:

| Tier | Embeds | Runs on | Use when |
|---|---|---|---|
| `text` | transcript | any laptop, CPU | spoken content is what matters |
| `visual` | transcript + keyframes | a modern laptop | slides and demos matter too |
| `full` | transcript + frame + video + audio + fused | a GPU | you want visual queries to work |

`reelreader doctor` reports which tier your machine can carry. See
[docs/SPEC.md](docs/SPEC.md) for the views, index format, and the MRL notes.

## Privacy

- No telemetry.
- The only outbound calls are to the chat and embedding endpoints you configure.
- API keys come from the environment (or the OS keychain with the `keyring`
  extra), never from the TOML file.
- `reelreader config` shows exactly what will be sent, with the key redacted.

## Documentation

- [docs/SPEC.md](docs/SPEC.md) — product and technical specification
- [docs/UX.md](docs/UX.md) — the three-pane loop, seekable citations, presenter window
- [docs/STORIES.md](docs/STORIES.md) — prioritised backlog with acceptance criteria
- [docs/adr/0001-python-core-tauri-sidecar.md](docs/adr/0001-python-core-tauri-sidecar.md) — why this stack
- [CONTEXT.md](CONTEXT.md) — vocabulary and invariants for contributors

## Development

```bash
uv venv && uv pip install -e '.[dev]'
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest
.venv/bin/reelreader doctor
```

## License

MIT. See [LICENSE](LICENSE).
