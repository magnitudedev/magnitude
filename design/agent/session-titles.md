---
applies_to:
  - packages/agent/src/projections/chat-title.ts
  - packages/agent/src/workers/chat-title-worker.ts
  - packages/agent/src/util/chat-title.ts
  - packages/agent/src/display-view/snapshot.ts
  - packages/agent/tests/chat-title.vitest.ts
---

# Session titles

## Policy

A session title is a deterministic derivation of its first nonsynthetic root user message. Forked
and synthetic messages do not participate in session naming.

Title derivation collapses every run of whitespace to one space, trims surrounding whitespace, and
keeps the first 50 Unicode code points. It does not change capitalization, preserve word boundaries,
or add an ellipsis. If the first eligible message has no text after normalization, the session keeps
the default title and later messages do not replace it.

Title derivation never invokes a model.

## Ownership and persistence

The title projection derives current display state directly from the existing user-message event.
It does not require a separate title event. A worker persists a nonempty derived title as session
metadata and updates trace metadata; persistence does not independently decide title content.

## Acceptance criteria

- The same first eligible message always produces the same title without provider access.
- At most one nonempty derived title is persisted for a session runtime.
- Forked, synthetic, and later user messages cannot change the title.
- Display state and persisted metadata use the same derivation.
