---
applies_to:
  - packages/acn-protocol/src/schemas/display.ts
  - packages/agent/src/display/**
  - packages/agent/src/display-view/**
  - packages/client-common/src/utils/root-detail.ts
  - cli/src/features/chat-timeline/**
  - cli/src/features/agent-status/**
  - web/src/components/chat-timeline.tsx
  - web/src/components/messages/**
  - web/src/components/inline-work-activity.tsx
  - web/src/components/general-settings.tsx
---

# Conversation timeline presentation

## Authority

Agent display projections own chronological conversation facts, lifecycle timing, typed tool
presentation, and semantic grouping. A client may omit facts from its rendered surface or choose a
different visual treatment, but it never reconstructs lifecycle or regroups raw events.

The presentation mode in a requested display shape selects ordinary versus transcript diagnostic
detail. It does not express a user's thinking-visibility preference. Thinking entries are available
in both modes; CLI normal mode and web client preferences independently decide whether to render
them.

## Thinking runs

A thinking run is the maximal contiguous sequence of provider thought activity without an
intervening visible tool, assistant response, interruption, or turn boundary. Multiple adjacent
provider thought blocks belong to one run. The run records authoritative start and completion
timestamps and whether it is active. Empty or explicitly suppressed runs do not remain in history.

CLI transcript renders thinking content inline. Web may render an expandable disclosure controlled
by a persisted client-only preference. Hiding thinking changes presentation only; it does not alter
events, projection state, accepted timeline ordering, or agent execution.

## Active work

Transient model residency and request progress remain authoritative agent/model state. Web renders
that state at the chronological conversation tail only while the root actor is working; slot
residency outside an active turn is not conversation activity. Indeterminate model loading never
receives a fabricated percentage or a determinate-looking bar. Only authoritative model-loading
progress uses a progress bar; request preparation and prefill remain compact textual activity with
their available token and cache detail. A specific activity takes precedence over a
generic one: model loading, prefill, visible thinking, a running tool, and assistant streaming do
not produce simultaneous duplicate work indicators.

Tool summaries preserve the agent presentation's grouping and shared label grammar. Completion is
represented by the tool phase and never by appending generic text such as "Done." Final assistant
completion uses the persisted work-summary message and retains its model and performance metadata.
After the authoritative work summary completes a user turn, web presents one copy and relative-time
footer after its final assistant response. No provisional footer follows streaming or intermediate
assistant messages. Copying the completed footer concatenates that turn's assistant prose and does
not create empty metadata rows between interleaved assistant responses and tool activity.

## File-tool navigation

Web file writes and edits are compact timeline records, not embedded editors or diff viewers. Their
relative path opens the Project Files surface only through the current cwd's explicitly registered
Project and the Project file client boundary. The renderer never creates a Project implicitly or
opens an absolute host path directly. Other clients may render the authoritative diff data.

## Conformance

- Adjacent thinking blocks remain one timed run; visible timeline boundaries split runs.
- Normal accepted timelines contain thinking entries even when a renderer hides them.
- Desktop thinking visibility is client presentation state and never ACN configuration.
- One specific live activity is visible at a time and final work summaries remain durable history.
- Tool grouping and labels are not reimplemented in React.
- Desktop file edit/write rows contain no inline diff and navigate only through Project Files.
