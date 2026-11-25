# Reelreader UX

The product is one loop: ask, read, seek. Everything else supports it.

## 1. The loop

```
ask ──▶ grounded answer ──▶ citation [12:34] ──▶ player seeks ──▶ transcript follows
  ▲                                                                    │
  └──────────────── select text or a moment to ask again ◀─────────────┘
```

A user must be able to go from a question to the spoken sentence in two clicks
and see the keyframe for it in one more.

## 2. Layout

```
┌────────────────┬───────────────────────────────────┬──────────────────────┐
│ Library        │ Player                            │ Chat                 │
│                │ ┌───────────────────────────────┐ │ ┌──────────────────┐ │
│ ● Week 3       │ │                               │ │ │ Q: What is a     │ │
│   ready        │ │          video                │ │ │    p-value?      │ │
│ ○ Week 4 42%   │ │                               │ │ │                  │ │
│ ○ Week 5      │ └───────────────────────────────┘ │ │ A: ... [12:34]   │ │
│                │ ▸ 02:10 ──●────────────── 58:12   │ │      [18:02]     │ │
│ + Add video    │ filmstrip  ▢ ▢ ▣ ▢ ▢ ▢ ▢ ▢        │ └──────────────────┘ │
│                │                                   │ [Ask about video ▾] │
│                │ transcript                        │                      │
│                │  12:30  and that is why the …     │                      │
│                │ ▸12:34  a p-value is the prob…    │                      │
│                │  12:41  so if we set alpha …      │                      │
└────────────────┴───────────────────────────────────┴──────────────────────┘
```

- **Library (left):** videos with status rings; add, remove, reindex. Width
  fixed, collapsible.
- **Stage (centre):** player, scrubber, filmstrip, synced transcript. The active
  line is the only thing that auto-scrolls.
- **Chat (right):** conversation, citations, scope selector (this video / whole
  library). Collapsible to give the stage full width.

Below 1100 px the chat becomes a tab beside the transcript, not a third column.

## 3. The citation contract

Every citation chip contains a timestamp and seeks on click. Hovering previews
the segment: its first line, a thumbnail, and the time range. Clicking does three
things in one action:

1. seeks the player to `start - 0.5 s` (a little lead-in),
2. highlights the transcript line,
3. flashes the matching filmstrip frame.

Citations that fail validation are never shown. If the model returns none, the
answer is replaced by the abstention state.

## 4. Answer states

| State | What the user sees |
|---|---|
| Retrieving | "Looking through 14 segments…" with a thin progress shimmer |
| Answering | streaming text; citations appear as they are resolved |
| Grounded | answer + citation chips |
| Partial | answer + "only covers part of your question" + the gap |
| Abstained | "That is not covered in this lecture. The closest moment is 12:34." + a seek button |
| Provider error | "Anthropic rejected the key. Check ANTHROPIC_API_KEY." + Retry / Settings |
| Offline, no local model | "This needs a model. Start Ollama or add a key." + link |

The abstention state is a feature, not an error. It should feel like a confident
"no", not a failure.

## 5. Moments, not only messages

The chat is not the only entry point. The user can act on the video directly:

- click a transcript line → seek
- select transcript text → **Explain**, **Simplify**, **Make a flashcard**, **Ask about this**
- right-click a moment → **Ask about this**, **Copy link to 12:34**, **Export clip**
- click a filmstrip frame → seek + show the frame large
- drag on the scrubber → chapter labels appear

Keyboard: `Space` play/pause, `←/→` ±5 s, `J/L` ±10 s, `K` pause, `/` focus
search, `⌘K` command palette, `⌘F` find in transcript, `Esc` close panel.

## 6. Processing

A job is visible from the moment it starts:

```
Week 4.mp4
  ✓ probe      ✓ transcode
  ✓ transcribe 1,842 words
  ● segment    38 of 214
  ○ embed      (text: 2 min, frame: 9 min)
  ○ index
```

- Cost and time are estimated before the job starts, from duration and tier.
- Stages are resumable; a crash returns to the last completed segment.
- A failed stage names the cause and the fix, not a stack trace.
- Jobs run in the sidecar; the UI never freezes.

## 7. Onboarding

First run, three questions and no account:

1. **Where should videos live?** Default `~/Reelreader`, changeable.
2. **What should answer questions?** Local (Ollama detected / install), or a key.
3. **How much should I index?** Text (fast), Slides too, Everything (GPU), with a
   one-line cost each.

The app tests the choice with a real one-sentence call before accepting it, and
shows exactly what will leave the machine.

## 8. Presenter window

Studying often means two screens or one projector. The presenter window is a
second window, not a second app:

- player and captions on the external display,
- chat and transcript in the main window,
- one shared playhead: seeking anywhere moves both,
- a "hand over" state so the audience sees the player, not the chat.

## 9. Empty and edge states

- **No videos:** one drop target and a line: "Drop a lecture to begin."
- **Video with no audio:** explain that transcription is impossible; offer the
  visual tier only.
- **Very long video (>3 h):** warn and suggest splitting; keep going if asked.
- **Moved library:** detect missing paths, offer "Locate folder", store relative
  paths where possible.
- **Index older than the model config:** "The embedding model changed. Rebuild
  the index (4 min)."

## 10. Words

- Say **moment** for a segment, **lecture** for the video, **transcript** for the
  text.
- "Not covered" beats "I could not find relevant information."
- Never say "RAG", "vector", "embedding", or "chunk" in the interface.
- Errors name the thing and the fix: "Ollama is not running at
  localhost:11434. Start it, or switch to a key."

## 11. Accessibility

- Full keyboard path to every action; visible focus.
- Captions in the player, not only in the transcript panel.
- Respect reduced motion; no auto-playing scrubs.
- Contrast checked in light and dark.
- Every citation has an accessible name: "Seek to 12 minutes 34 seconds".
