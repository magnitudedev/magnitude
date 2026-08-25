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

A bound model has one ordered stream of provider-neutral `ResponseStreamEvent`s. A provider may
also expose request-local preparation through the optional
`ModelPreparationObserver<TPreparation>` argument to `stream`. The observer payload is generic and
consumer-defined; the AI contract defines no loading, queue, prefill, or local-inference activity
vocabulary. `TPreparation = never` is the default for providers without that capability.

Preparation observation is a side channel of the same request, not model output. Starting,
streaming, and ending remain caller-owned observations rather than provider events.

## Persistence boundary

Model activity is transient and must never enter the canonical harness or agent event log. The
agent model decorator supplies the preparation observer and projects its updates through runtime
Ambient state. The decorated model exposes only `ResponseStreamEvent`s to the harness, so only
response-derived `HarnessEvent`s can cross the persistence boundary.

The agent derives its transient `Streaming` state from the first non-terminal
`ResponseStreamEvent`. A terminal response without semantic output never enters `Streaming`.

ICN reports requested progress chunks through the observer before filtering them from
provider-neutral response decoding. ICN-specific preparation data never enters
`ResponseStreamEvent` or a terminal payload.

## Guarantees

- Preparation callbacks run in source order before the corresponding semantic response chunks.
- A provider error remains a terminal response event, not activity.
- Stream completion, failure, or interruption clears transient activity.
- Consumers may omit the preparation observer without changing inference behavior.
- Tests must prove both activity projection and exclusion from harness output.
