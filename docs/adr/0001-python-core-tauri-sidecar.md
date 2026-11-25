# 0001 — Python engine, Tauri shell, one HTTP boundary

Date: 2026-11-25
Status: accepted

## Context

Reelreader must ship as a PyPI CLI and a desktop app. The work is media
processing (ffmpeg), speech recognition, local model inference, and vector
search. That ecosystem is Python-first: `faster-whisper`, `scenecut` libraries,
`sentence-transformers`, `fastembed`, `numpy`, `sqlite-vec`.

The fleet already contains both shapes: `slide-stream` (Python core, Typer CLI,
FastAPI, Tauri desktop) and `debrief` (Electron with a Python sidecar). The
`app-hq` decision tree says Tauri by default when OS integration is needed and
Electron when the Node ecosystem is unavoidable.

## Options

1. **Pure TypeScript** — one runtime, Electron only. Reimplements ASR and
   embedding glue, or shells out to Python anyway. Rejected: it would fork the
   engine from the CLI.
2. **Electron shell with a Python sidecar** — proven in `debrief`, reuses React
   code from `insight-lens`. Rejected: heavier binary and a second runtime for a
   UI that is mostly video, a list, and chat.
3. **Python engine + FastAPI + Tauri shell** — one engine, three surfaces, small
   shell, matches `slide-stream`. Accepted.
4. **Python engine only, no desktop** — fails the student audience.

## Decision

- The engine is a Python package, `reelreader`, published to PyPI.
- `reelreader serve` exposes the engine over HTTP on `127.0.0.1`.
- The desktop app is a Tauri v2 shell that spawns that server as a sidecar with a
  per-launch bearer token and an ephemeral port.
- The CLI and the desktop call the same engine; the desktop never reimplements
  retrieval.

## Consequences

Positive:

- One implementation of segmentation, retrieval, and grounding.
- The CLI is usable on day one and is the test harness for the desktop.
- Python media and model libraries are available without FFI work.
- The API is a stable integration point for scripts and future surfaces.

Negative:

- Two runtimes to package and sign (Tauri binary + Python sidecar).
- The sidecar must be bundled or the user must have Python; a bundled build is
  larger and platform-specific.
- Sidecar lifecycle (start, health, crash, orphan) is real work, specified in
  SPEC §16.
- Startup is slower than a single-process app; mitigated by a splash and a fast
  `/health`.

## Follow-ups

- Decide the sidecar packaging (PyInstaller vs a bundled interpreter) in M2.
- Keep the HTTP schema versioned so the shell and engine can update separately.
- Revisit only if the sidecar proves unshippable on Windows; the fallback is
  Electron with the same API, not a rewrite of the engine.
