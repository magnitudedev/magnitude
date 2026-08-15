---
applies_to:
  - packages/client-common/src/state/notification-area-state.ts
  - cli/src/app.tsx
  - cli/src/index.tsx
  - cli/src/platform/effect-logger.ts
  - cli/src/features/notification-area/**
  - cli/src/features/composer/**
  - cli/src/features/model-menus/**
  - cli/src/features/local-inference/footer-status.tsx
---

# CLI notification area

The CLI has one footer notification area shared by the composer and the model-menu surface. Exactly
one notification is visible. Switching surfaces changes only its layout; it does not create a second
notification state or reset client-owned notifications.

## Sources and ownership

A persistent notification is a pure projection of authoritative state. Download activity is
derived from local-model acquisition state, and selected-model memory guidance is derived from the
selected local model's current-headroom state. Selected-model `Requested` residency is likewise
projected as activity, and `Failed` residency is projected as an error. None of these facts is
copied into writable client state, started by a UI event, or cleared by a timer.

An ephemeral notification represents a transient client event. Shared client state owns its exact
identity and lifetime in a retained Effect Atom. Publishing adds one identified occurrence;
dismissal, including timed dismissal, removes that exact occurrence. Concurrent timers cannot
dismiss a different notification, and changing between the composer and model menu cannot lose an
occurrence.

The area stores semantic priority and action values, never terminal colors, callbacks, server
state, or duplicated lifecycle counters. Terminal rendering owns layout, theme colors, hover state,
and action dispatch.

## Resolution

The active notification is derived from all current persistent projections and ephemeral
occurrences. Priority is:

```text
error > warning > notice > activity
```

The newest occurrence wins within one priority. Removing the active occurrence reveals the next
current notification; ongoing activity therefore reappears automatically. A catalog action remains
attached to model-download activity rather than being inferred from its message.

## Memory guidance

When the selected local model's authoritative current-headroom state is `Insufficient`, the area
shows `! Low memory: close memory-intensive apps (need X.Y GB) to load model` at warning priority
and in the warning theme color. `X.Y` is the server-published minimum additional availability,
formatted to one decimal GB. When the composer footer stacks into two rows, the same notification
uses the compact presentation `! Low memory: Free X.Y GB to load`. The warning remains visible while
that condition remains true. It is not an ephemeral selection acknowledgement and has no dismissal
timer. It clears only when the selected model changes or fresh local-model state no longer reports
insufficient headroom.

## Conformance

- The composer and model menu render the same resolved notification.
- Model downloads have one derived count and no client-retained download state.
- Low-memory guidance has one derived condition and no selection-handler side channel.
- Requested and failed residency are derived from the authoritative selected-slot state.
- Only ephemeral client events enter writable notification state.
- Warning and error occurrences take precedence over ongoing downloads.
- No notification producer writes directly to a terminal component or a parallel toast store.
