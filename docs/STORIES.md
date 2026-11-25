# Reelreader stories

Ordered by dependency, then by user value. Each story is small enough for one
sitting and ends with something a person can use. Acceptance criteria are the
definition of done.

Status: M0 complete. Everything below is open.

## M0 — Skeleton

### 1. Repository, config, providers, doctor — done

- [x] `reelreader` installs from the repo and runs `--version`.
- [x] `init` writes a TOML and creates the library.
- [x] `doctor` reports Python, library, config, chat model, embedding model,
      tier, ffmpeg, and optional extras.
- [x] OpenAI-compatible chat works with and without a key.
- [x] Anthropic chat works with a key and fails clearly without one.
- [x] `config` never prints a key.
- [x] Anthropic is rejected as an embedding provider with an explanation.
- [x] `ruff` and `pytest` green; CI calls the fleet workflow.

### 2. Library database and job runner — done

- [x] `library.db` is created on first use with the tables in SPEC §8.
- [x] A job has a stage, state, and error, and survives a process restart.
- [x] `reelreader list` shows videos and their job state.
- [x] Reopening an existing database runs no duplicate migrations.

## M1 — The text loop

### 3. Ingest a file or URL — done

- [x] `reelreader add lecture.mp4` references the file in place, records the
      absolute path, and computes `source_sha256`; `--copy` brings a copy in.
- [x] Adding the same file twice is a no-op with a clear message.
- [x] `--title` overrides the derived title.
- [x] A video with no audio is accepted but marked `no_audio`.
- [x] A folder argument ingests every video inside it, recursively.
- [x] `yt-dlp` is used for URLs when available; a clear error otherwise.

### 4. Transcribe and segment

- [ ] `reelreader index` runs probe → transcode → transcribe → segment.
- [ ] Output matches SPEC §7.3 and §7.4 exactly.
- [ ] Segments never cross a scene change when `scenedetect` is installed.
- [ ] Re-running resumes at the first missing artifact.
- [ ] A 10-minute sample indexes in under 5 minutes on CPU.
- [ ] Interrupting with Ctrl-C leaves a resumable library.

### 5. Text index

- [ ] Each segment is embedded with the configured embedder.
- [ ] `manifest.json` records model, revision, dim, normalisation, and count.
- [ ] A dimension mismatch between config and manifest refuses with a rebuild
      command.
- [ ] Cosine search returns the expected segment for 10 hand-written questions.

### 6. Search without a model

- [ ] `reelreader search "p-value"` prints time, score, and text.
- [ ] `--view` selects text, frame, video, audio, or fused.
- [ ] Hybrid ranking (BM25 + dense, RRF k=60) beats dense alone on the eval set.
- [ ] `--json` emits one object per hit.

### 7. Ask with seekable citations

- [ ] `reelreader ask "what is a p-value?"` returns an answer with `[mm:ss]`
      citations.
- [ ] Every citation maps to a retrieved segment; invalid ones are repaired or
      cause abstention.
- [ ] With no relevant segment, the reply is the abstention message and the
      nearest moment.
- [ ] `--json` returns `{answer, citations, segments, trace}`.
- [ ] A test stubs the provider and asserts the prompt contains only retrieved
      text.

## M2 — Desktop

### 8. Tauri shell and sidecar lifecycle

- [ ] `npm run tauri dev` opens a window and starts the sidecar.
- [ ] The sidecar binds `127.0.0.1` on an ephemeral port with a per-launch token.
- [ ] Closing the window kills the sidecar and any children (verified by PID).
- [ ] A crashed sidecar shows a restart action, not a blank window.
- [ ] The webview loads only bundled assets; CSP blocks remote script.

### 9. Three-pane UI

- [ ] Library, player, chat, and synced transcript render as in docs/UX.md.
- [ ] Clicking a citation seeks the player and highlights the transcript line.
- [ ] The active transcript line scrolls into view; manual scroll pauses that for
      3 seconds.
- [ ] Keyboard shortcuts from UX §5 work.
- [ ] Light and dark both meet contrast checks.

### 10. Export

- [ ] `export --format md` writes a study guide with timestamps.
- [ ] `srt` and `vtt` are valid (checked by a parser in tests).
- [ ] `anki` writes a two-column CSV importable by Anki.
- [ ] Export works from CLI and from a desktop menu item.
- [ ] No export requires the app to open the file again.

## M3 — Visual

### 11. Local HTTP API

- [ ] Endpoints in SPEC §15 exist with the documented error codes.
- [ ] `/events` streams stage progress over SSE.
- [ ] Every request without the launch token is rejected.
- [ ] A path traversal attempt against `/media` is rejected and logged.

### 12. Evaluation harness

- [ ] `reelreader eval` reads `manual.jsonl` and reports recall@5 and MRR.
- [ ] Results are broken down by view and tier.
- [ ] Synthetic and manual queries are reported separately.
- [ ] The command exits non-zero when recall drops more than 2 points from the
      recorded baseline.

### 13. Keyframes, filmstrip, frame tier

- [ ] Keyframes are extracted at segment midpoint and at scene changes.
- [ ] The filmstrip shows one frame per segment and seeks on click.
- [ ] `tier = "visual"` embeds frames and improves visual-query recall over text.
- [ ] `doctor` estimates index time for the tier before the job starts.

## M4 — Full multimodal

### 14. Audio, video, fused views

- [ ] Clip and wav artifacts exist for every segment.
- [ ] `tier = "full"` produces text, frame, video, audio, and fused matrices.
- [ ] fp16 is refused or warned; vectors are checked for NaN.
- [ ] MRL truncation to 256 renormalises and keeps ≥90% of recall@5.
- [ ] A GPU-less machine gets a clear estimate and can queue the job.

### 15. Presenter window

- [ ] A second window shows player and captions only.
- [ ] Seeking in either window moves both.
- [ ] The audience view never shows chat text.

## M5 — Ship

### 16. Packaging and release

- [ ] `uv build` produces a wheel and sdist; `uv tool install` works from PyPI
      test index.
- [ ] macOS build is signed and notarised with the fleet API key.
- [ ] Windows and Linux builds are produced by CI.
- [ ] The installer does not bundle model weights whose licence forbids it;
      first-run download is explicit.
- [ ] README states the privacy posture, data location, and how to export.
- [ ] The app is registered in app-hq `registry/apps.yaml`.
