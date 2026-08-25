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

A bound model returns one ordered `ModelStreamEvent<TPreparation>` stream. Ordinary events remain
provider-neutral `ResponseStreamEvent`s. A provider may additionally emit the explicitly identified
`preparation_update` event carrying a generic consumer-defined preparation payload. The AI contract
defines no loading, queue, prefill, or local-inference preparation vocabulary.
`TPreparation = never` is the default for providers without that capability.

`Starting` is represented structurally by the pending `stream` Effect before it returns the
`ModelStreamResult`. Ending remains part of the terminal response event or Effect interruption.

## Persistence boundary

Model activity is transient and must never enter the canonical harness or agent event log. The
agent model decorator observes the raw mixed stream in source order, projects activity through
runtime Ambient state, and partitions it into preparation and response branches. It drains the
preparation branch and exposes only the response branch. The harness therefore accepts only
response events and contains no preparation-specific behavior.

The agent derives its transient `Streaming` state from the first non-terminal
`ResponseStreamEvent`. A terminal response without semantic output never enters `Streaming`.

ICN maps requested progress chunks to `preparation_update` events before the corresponding response
events. ICN-specific preparation data never enters `ResponseStreamEvent` or a terminal payload.

## Guarantees

- Preparation events occur in source order before the corresponding semantic response events.
- A provider error remains a terminal response event, not activity.
- Stream completion, failure, or interruption clears transient activity.
- Agent-model consumers receive only response semantics without filtering.
- Tests must prove both activity projection and exclusion from the returned agent-model stream.
