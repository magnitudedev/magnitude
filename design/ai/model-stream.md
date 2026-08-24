---
applies_to:
  - packages/ai/src/model/**
  - packages/harness/src/turn/harness.ts
  - packages/agent/src/model/agent-model.ts
  - packages/agent/src/model/model-request-activity.ts
  - packages/icn/src/provider/**
---

# Model stream

## Contract

A bound model has one ordered, ephemeral output stream. The stream may contain provider-neutral
response events and generic provider activity:

- `preparation_update<TPreparation>` reports provider-owned preparation state; and
- `ResponseStreamEvent` carries semantic output and the terminal result.

`TPreparation = never` removes preparation updates from providers without that capability. The
generic AI contract defines no loading, queue, prefill, or local-inference activity vocabulary.

The model API has no lifecycle observer or callback output. Starting and ending are caller-owned
observations, not provider stream events.

## Persistence boundary

Model activity is transient and must never enter the canonical harness or agent event log. The
agent model decorator observes activity from the ephemeral model stream and projects it through
runtime Ambient state. The harness exhaustively removes activity before response dispatch; only
response-derived `HarnessEvent`s may cross the persistence boundary.

The agent derives its transient `Streaming` state from the first non-terminal
`ResponseStreamEvent`. A terminal response without semantic output never enters `Streaming`.

ICN converts requested progress chunks into generic model activity before provider-neutral response
decoding. ICN-specific preparation data never enters `ResponseStreamEvent` or a terminal payload.

## Guarantees

- Activity and response ordering is preserved within one stream.
- A provider error remains a terminal response event, not activity.
- Stream completion, failure, or interruption clears transient activity.
- Direct response consumers may ignore activity without changing inference behavior.
- Tests must prove both activity projection and exclusion from harness output.
