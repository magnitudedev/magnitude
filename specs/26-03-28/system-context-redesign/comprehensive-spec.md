# Comprehensive Implementation Specification — System Context Redesign

## Table of Contents

1. [Overview](#overview)
2. [Architecture Overview](#architecture-overview)
3. [Output Format Specification](#output-format-specification)
4. [Type Architecture](#type-architecture)
5. [Composition Layer (`context-composer.ts`)](#composition-layer-context-composerts)
6. [Orchestration Layer (`memory.ts`)](#orchestration-layer-memoryts)
7. [Rendering Layer (`context-stream.ts` + `tool-call-render.ts` + `results.ts`)](#rendering-layer-context-streamts--tool-call-renderts--resultsts)
8. [Subagent Activity Enrichment](#subagent-activity-enrichment)
9. [Replay and Hydration](#replay-and-hydration)
10. [Legacy Removal](#legacy-removal)
11. [Consumer Migration](#consumer-migration)
12. [Behavioral Changes Register](#behavioral-changes-register)
13. [Implementation Plan (5 steps)](#implementation-plan-5-steps)
14. [Acceptance Checklist](#acceptance-checklist)

---

## Overview

This redesign fully replaces the legacy lead-context pipeline (`system_inbox` + `comms_inbox` + status observable) with a single typed `context_inbox` model composed of a results lane and a chronological timeline lane, rendered to the lead using the design-spec output format (flat time markers, in-flow agent blocks, optional factual attention section).

The redesign exists because current behavior is fragmented across dual inboxes, mixed abstraction boundaries in `memory.ts` (event semantics + queue orchestration + formatting), and legacy wrappers/rosters that conflict with the target output model and create maintenance complexity.

Hard constraints:
- Full replacement only: no dual-path runtime, no legacy fallback, no compatibility shims left after migration.
- Only two intentional behavioral changes:
  1. New lead context format.
  2. Removal of global agent status roster (`agents_status`).
- One accepted ordering difference may occur due to unifying queues into single timeline ordering (see [Behavioral Changes Register](#behavioral-changes-register)).

---

## Architecture Overview

### Three-layer pipeline

1. **Composer (`context-composer.ts`)**  
   Converts raw events/signals into typed IR entries (`ResultEntry`, `TimelineEntry`).

2. **Memory orchestration (`memory.ts`)**  
   Owns queue lifecycle, flush timing, coalescing, mention patching, and persisted message history.

3. **Renderer (`context-stream.ts`)**  
   Converts typed lanes into `ContentPart[]` following the output format spec.

### Data flow diagram (text)

```text
Persisted AppEvent stream
  -> projection handlers + signal handlers
  -> context-composer.ts
      -> ResultEntry | TimelineEntry
  -> memory.ts queue (QueuedEntry[lane=result|timeline])
  -> turn_started flush
      -> Message { type: 'context_inbox', results, timeline }
  -> transformMessage()
      -> formatContextInbox({ results, timeline, timezone })
      -> ContentPart[] consumed by lead model
```

### Abstraction boundaries and ownership

- **Composer owns semantics mapping** (what happened).
- **Memory owns buffering/flush/history** (when it is shown).
- **Renderer owns presentation** (how it is shown).
- Memory must not generate XML payload strings for context formatting.

---

## Output Format Specification

This section is normative and must match `design-spec.md`.

### Canonical rendering order

1. Results lane (if non-empty)
2. Chronological lane (if non-empty)
3. `<attention>` section at end only when useful

No outer wrapper tags (`<system>`, `<comms>`, `<results>`).

### Results lane

Render lead's prior-turn tool results first, flat, preserving multimodal parts.

Example:

<read path="src/auth.ts">...</read>
<edit path="src/auth.ts">Applied successfully</edit>
<shell>
<stdout>Tests: 23 passed</stdout>
<exitCode>0</exitCode>
</shell>

### Chronological lane

Render minute-separated timeline using flat markers:

```text
--- 02:20 ---
```

All entries until next marker belong to that minute window.

### Agent block rules

<agent id="builder-auth" role="builder" status="working|idle">
plain thought line
<read path="src/auth.ts"/>
<message to="lead">...</message>
<idle/>
</agent>

Rules:
- Structured activity lines are XML (`tool`, `message`, `idle`, `error`).
- Thoughts render as plain text lines.
- Agent blocks split across minute boundaries.
- Agent appears only when new unseen activity or new state transition exists.

### Tool-call rendering primitive

- Generic compact XML renderer.
- Keep tag and attrs.
- Optional truncated body.
- No per-tool display branching.

### Non-agent event rendering

Examples (must be supported):

<message from="user">...</message>
<user-to-agent agent="builder-auth">...</user-to-agent>
<user-presence>...</user-presence>
<file path="src/auth.ts">...</file>
<file-update path="package.json">@@ ...</file-update>
<phase_criteria name="deploy-config" status="passed" type="agent" agent="builder-deploy"/>
<workflow_phase name="deployment" phase="2/3">Set up CI/CD pipeline.</workflow_phase>
<reminder>Review the builder's work...</reminder>

### Attention section

At end only; factual bullets only:

<attention>
- user message at 02:27
- builder-db went idle at 02:24
</attention>

No interpretation, no full-content duplication.

### Time marker rules

1. First chronological entry always emits marker.
2. Emit marker on minute change.
3. Default marker: `--- HH:MM ---`
4. Full date marker when full date has not been rendered for >10 minutes: `--- YYYY-MM-DD HH:MM ---`
5. Markers are separators, not containers.

### Progressive activation

- User-only simple chat -> user message only.
- Results-only -> results lane only.
- Add chronological lane only if non-result entries exist.
- Add attention only for burial risk.

### Explicit exclusions

- No idle roster element.
- No background-process section.
- No system/comms/results wrappers.
- No nested time windows.

---

## Type Architecture

All type definitions below are complete and authoritative.

### Message union (5 variants)

```ts
import type { ContentPart, ImageMediaType } from '../content'
import type { ResponsePart, StrategyId, TurnToolCall, ObservedResult } from '../events'

export type Message =
  | {
      type: 'session_context'
      source: 'system'
      content: ContentPart[]
    }
  | {
      type: 'assistant_turn'
      source: 'agent'
      content: ContentPart[]
      strategyId: StrategyId
      responseParts: readonly ResponsePart[]
    }
  | {
      type: 'compacted'
      source: 'system'
      content: ContentPart[]
    }
  | {
      type: 'fork_context'
      source: 'system'
      content: ContentPart[]
    }
  | {
      type: 'context_inbox'
      source: 'system'
      results: readonly ResultEntry[]
      timeline: readonly TimelineEntry[]
    }
```

### ResultEntry union (all variants)

```ts
export type ResultEntry =
  | {
      kind: 'tool_results'
      toolCalls: readonly TurnToolCall[]
      observedResults: readonly ObservedResult[]
      error?: string
    }
  | {
      kind: 'interrupted'
    }
  | {
      kind: 'error'
      message: string
    }
  | {
      kind: 'noop'
    }
```

### TimelineEntry union (all variants, full fields)

```ts
export type TimelineEntry =
  | {
      kind: 'user_message'
      timestamp: number
      text: string
      attachments: readonly TimelineAttachment[]
    }
  | {
      kind: 'user_to_agent'
      timestamp: number
      agentId: string
      text: string
    }
  | {
      kind: 'agent_block'
      timestamp: number
      firstAtomTimestamp: number
      lastAtomTimestamp: number
      agentId: string
      role: string
      atoms: readonly AgentAtom[]
    }
  | {
      kind: 'subagent_user_killed'
      timestamp: number
      agentId: string
      agentType: string
    }
  | {
      kind: 'user_presence'
      timestamp: number
      text: string
      confirmed: boolean
    }
  | {
      kind: 'file'
      timestamp: number
      path: string
      content: string
    }
  | {
      kind: 'file_update'
      timestamp: number
      path: string
      text: string
    }
  | {
      kind: 'workflow_phase'
      timestamp: number
      name?: string
      phase?: string
      text: string
    }
  | {
      kind: 'phase_criteria'
      timestamp: number
      payload: PhaseCriteriaPayload
    }
  | {
      kind: 'phase_verdict'
      timestamp: number
      passed: boolean
      verdictText: string
      workflowCompleted: boolean
      nextPhase?: {
        name?: string
        phase?: string
        prompt: string
      }
    }
  | {
      kind: 'skill_started'
      timestamp: number
      skillName: string
      firstPhase?: string
      prompt: string
    }
  | {
      kind: 'skill_completed'
      timestamp: number
      skillName: string
    }
  | {
      kind: 'reminder'
      timestamp: number
      text: string
    }
  | {
      kind: 'observation'
      timestamp: number
      parts: readonly ContentPart[]
    }
```

### AgentAtom union (all variants)

```ts
export type AgentAtom =
  | {
      kind: 'thought'
      timestamp: number
      text: string
    }
  | {
      kind: 'tool_call'
      timestamp: number
      toolCallId: string
      tagName: string
      attributes: Record<string, string>
      body?: string
      status: 'success' | 'error' | 'interrupted'
      exitCode?: number
      error?: string
    }
  | {
      kind: 'message'
      timestamp: number
      direction: 'to_lead' | 'from_user' | 'from_lead'
      text: string
    }
  | {
      kind: 'error'
      timestamp: number
      message: string
    }
  | {
      kind: 'idle'
      timestamp: number
      reason?: 'stable' | 'interrupt' | 'error'
    }
```

Note on target representations:
- These types represent the target architecture.
- Kinds such as `agent_block`, `phase_criteria` (typed payload), and `user_to_agent` are new representations that replace legacy equivalents or introduce new capability per design spec.

### PhaseCriteriaPayload union

```ts
export type PhaseCriteriaPayload =
  | {
      source: 'agent'
      name: string
      status: 'passed' | 'failed' | 'pending'
      agentId: string
      reason?: string
    }
  | {
      source: 'shell'
      name: string
      status: 'passed' | 'failed' | 'pending'
      command: string
      reason?: string
    }
  | {
      source: 'user'
      name: string
      status: 'passed' | 'failed' | 'pending'
      reason?: string
    }
```

### TimelineAttachment union

```ts
export type TimelineAttachment =
  | {
      kind: 'image'
      mediaType: ImageMediaType
      base64: string
      width?: number
      height?: number
    }
  | {
      kind: 'mention'
      path: string
      contentType: 'text' | 'image' | 'directory'
      resolved?: {
        content?: string
        error?: string
        truncated?: boolean
        originalBytes?: number
        image?: {
          mediaType: ImageMediaType
          base64: string
        }
      }
    }
```

### QueuedEntry type

```ts
export type QueuedEntry =
  | {
      lane: 'result'
      timestamp: number
      seq: number
      entry: ResultEntry
      coalesceKey?: string
    }
  | {
      lane: 'timeline'
      timestamp: number
      seq: number
      entry: TimelineEntry
      coalesceKey?: string
    }
```

### ForkMemoryState (new shape)

```ts
export interface ForkMemoryState {
  messages: readonly Message[]
  queuedEntries: readonly QueuedEntry[]
  currentTurnId: string | null
  currentChainId: string | null
  pendingPresenceText: string | null
  nextQueueSeq: number
}
```

---

## Composition Layer (`context-composer.ts`)

### Dependencies interface

```ts
export interface ComposeContextDeps {
  resolveAgentByForkId(
    forkId: string
  ): { agentId: string; role: string; parentForkId: string | null } | null
}
```

### Required function signatures

```ts
export function toResultToolResults(args: {
  toolCalls: readonly TurnToolCall[]
  observedResults: readonly ObservedResult[]
  error?: string
}): ResultEntry | null

export function toResultInterrupted(): ResultEntry
export function toResultError(args: { message: string }): ResultEntry
export function toResultNoop(): ResultEntry

export function toTimelineUserMessage(args: {
  timestamp: number
  text: string
  attachments: readonly TimelineAttachment[]
}): TimelineEntry

export function toTimelineUserToAgent(args: {
  timestamp: number
  agentId: string
  text: string
}): TimelineEntry

export function toTimelineAgentBlock(args: {
  timestamp: number
  firstAtomTimestamp: number
  lastAtomTimestamp: number
  agentId: string
  role: string
  atoms: readonly AgentAtom[]
}): TimelineEntry

export function toTimelineSubagentUserKilled(args: {
  timestamp: number
  agentId: string
  agentType: string
}): TimelineEntry

export function toTimelineUserPresence(args: {
  timestamp: number
  text: string
  confirmed: boolean
}): TimelineEntry

export function toTimelineFileFirstMentioned(args: {
  timestamp: number
  path: string
  content: string
}): TimelineEntry

export function toTimelineFileUpdate(args: {
  timestamp: number
  path: string
  text: string
}): TimelineEntry

export function toTimelineWorkflowPhase(args: {
  timestamp: number
  name?: string
  phase?: string
  text: string
}): TimelineEntry

export function toTimelinePhaseCriteria(args: {
  timestamp: number
  payload: PhaseCriteriaPayload
}): TimelineEntry

export function toTimelinePhaseVerdict(args: {
  timestamp: number
  passed: boolean
  verdictText: string
  workflowCompleted: boolean
  nextPhase?: { name?: string; phase?: string; prompt: string }
}): TimelineEntry

export function toTimelineSkillStarted(args: {
  timestamp: number
  skillName: string
  firstPhase?: string
  prompt: string
}): TimelineEntry

export function toTimelineSkillCompleted(args: {
  timestamp: number
  skillName: string
}): TimelineEntry

export function toTimelineReminder(args: {
  timestamp: number
  text: string
}): TimelineEntry

export function toTimelineObservation(args: {
  timestamp: number
  parts: readonly ContentPart[]
}): TimelineEntry
```

### Responsibility

- Pure mapping only.
- No XML construction for output formatting.
- No queue mutation.
- No message history mutation.

---

## Orchestration Layer (`memory.ts`)

### Complete handler mapping table

| Source | Mapping |
|---|---|
| `session_initialized` | Preserve behavior via equivalent typed representation: `session_context` |
| `oneshot_task` | Preserve oneshot task behavior via equivalent typed representation in `session_context` |
| `user_message` | Preserve current conditional via equivalent typed representation: immediate append to persisted `context_inbox.timeline` only when `forkId === null` and no active turn; otherwise queue `timeline.user_message`. **NEW:** when `user_message` targets a subfork (`event.forkId !== null`), also enqueue parent-fork `timeline.user_to_agent` so lead sees directed user-to-agent traffic. |
| `skill_activated` (user source) | Preserve same conditional as `user_message` via equivalent typed representation: immediate append only when root fork and no active turn; otherwise queue `timeline.user_message` with slash text |
| `file_mention_resolved` | Preserve mention-resolution behavior via equivalent typed representation: patch timeline user attachments (persisted + queued) |
| `turn_started` | Preserve queue lifecycle behavior via equivalent typed representation: flush queued entries into one `context_inbox`; apply presence injection/noop guard |
| `tool_event` | No-op |
| `observations_captured` | Immediate append `timeline.observation` to persisted `context_inbox` |
| `turn_completed` | Preserve turn-completion behavior via equivalent typed representation: append `assistant_turn`; queue `result.tool_results`/`result.interrupted`; queue reminder/workflow timeline as needed |
| `turn_unexpected_error` | Immediate append `result.error` to persisted `context_inbox`; clear current turn id |
| `skill_started` | Queue `timeline.skill_started` |
| `phase_criteria_verdict` | Queue `timeline.phase_criteria` |
| `phase_verdict` | Queue `timeline.phase_verdict` via equivalent typed representation; renderer must preserve workflow suffix output behavior. Current memory builds `<phase_verdict>` XML with inline suffix (`<workflow_completed/>` or `<workflow_phase>` text). New model stores typed fields (`passed`, `verdictText`, `workflowCompleted`, `nextPhase`); renderer must produce equivalent XML output from these fields. Parity test required. |
| `skill_completed` | Queue `timeline.skill_completed` |
| `interrupt` | No-op |
| `compaction_completed` | Preserve compaction rewrite behavior via equivalent typed representation |
| `agent_created` (global) | Initialize new fork with empty message history (spawn-only; clone mode exists in event types but is not used in current agent creation). Preserve optional `fork_context` message injection via equivalent typed representation. Queue `parentOnSpawn` lifecycle reminder in parent fork. |
| `OutboundMessages.messageCompleted` | Preserve target-fork routing behavior via equivalent typed representation: resolve sender agent id (`lead` for root sender, agent id for subfork sender), ignore `dest === 'user'`, and queue message atom into the resolved target fork's `timeline.agent_block` path |
| `FileAwareness.fileFirstMentioned` | Queue `timeline.file` |
| `FileAwareness.fileUpdateNotification` | Queue `timeline.file_update` with coalesce key `file-update:${path}` |
| `SubagentActivity.unseenActivityAvailable` | Preserve subagent activity signaling behavior via equivalent typed representation: queue `timeline.agent_block` entries |
| `AgentStatus.agentBecameIdle` | Preserve idle transition behavior via equivalent typed representation: queue terminal idle atom in `agent_block` + optional lifecycle reminder |
| `AgentStatus.subagentUserKilled` | Preserve subagent-killed notification behavior via equivalent typed representation: queue `timeline.subagent_user_killed` |
| `UserPresence.userReturnedAfterAbsence` | Preserve deferred presence behavior via equivalent typed representation: set root `pendingPresenceText` injected on next `turn_started` |

### Queue model and flush mechanics

- Queue only `QueuedEntry`.
- Sequence key `(timestamp, seq)` for stable deterministic ordering.
- Flush on `turn_started`:
  1. Inject deferred presence if root fork.
  2. Partition queue into results/timeline.
  3. Stable sort by `(timestamp, seq)`.
  4. Emit one `context_inbox` message with both lanes.
  5. If lanes empty and last persisted message is `assistant_turn`, append `ResultEntry.noop`.
  6. Clear queue and set active turn metadata.

### Immediate append mechanics

- `turn_unexpected_error` -> immediate persisted append to latest mergeable `context_inbox.results`.
- `observations_captured` -> immediate persisted append to latest mergeable `context_inbox.timeline`.
- Purpose: preserve durability even if no subsequent `turn_started`.

### Mention resolution contract

- Match user message by `sourceMessageTimestamp`.
- Match mention by `contentType:path`.
- Patch both:
  - persisted `messages[].context_inbox.timeline.user_message.attachments`
  - queued timeline user messages
- Copy-on-write; no-op if no target.

### Coalescing policy

- File updates only use coalescing key `file-update:${path}`.
- New queued file update replaces prior queued file update for same path.
- No other timeline/result kinds coalesce by default.

### Noop guard behavior

- If flush produces no results/timeline and last persisted message source is `agent`, emit `result.noop` in new `context_inbox`.

### Deferred presence injection

- Root fork only.
- If explicit `pendingPresenceText` exists, inject as `timeline.user_presence`.
- Else if user currently absent, inject default absence presence text.
- Injected at beginning of timeline for that flush.

---

## Rendering Layer (`context-stream.ts` + `tool-call-render.ts` + `results.ts`)

### `formatContextInbox` signature and behavior

```ts
export interface FormatContextInboxInput {
  results: readonly ResultEntry[]
  timeline: readonly TimelineEntry[]
  timezone: string | null
}

export function formatContextInbox(input: FormatContextInboxInput): ContentPart[]
```

Behavior:
1. Render results lane first via `formatResults`.
2. Render timeline lane second with markers and agent grouping.
3. Synthesize `<attention>` last when qualifying events exist and burial risk threshold met.
4. Return multimodal `ContentPart[]`.

### Time marker rules

- First rendered timeline item always preceded by marker.
- Marker emitted when minute bucket changes.
- Marker format:
  - default `--- HH:MM ---`
  - full-date `--- YYYY-MM-DD HH:MM ---` when >10 minutes since last full-date marker.

### Agent block grouping and minute-boundary splitting

- Group consecutive agent atoms by `agentId` inside same minute window.
- Split block when:
  1. Minute boundary changes.
  2. Non-agent timeline event interrupts sequence.
  3. Agent id changes.
- Agent status:
  - `idle` if final atom in rendered block is idle.
  - otherwise `working`.

### Attention synthesis rules

Qualifying events:
- `timeline.user_message`
- `agent_block` terminal `idle` atom
- `agent_block` `error` atom

Burial-risk policy:
- Emit attention when qualifying events risk being missed due to subsequent content volume.
- Exact threshold is implementation-tuned; the design spec intentionally requires attention only when useful.

Bullets are factual timestamp pointers only.

### Progressive activation rules

- Empty results + single user_message with no additional events -> render only that message (no extra scaffolding).
- Results-only -> render only results lane.
- No attention when no qualifying buried items.
- Precedence exception: progressive activation overrides timeline marker rules; for simple user-only rendering, emit no time marker.

### `renderCompactToolCall` specification

```ts
export interface CompactToolCallInput {
  tagName: string
  attributes: Record<string, string>
  body?: string
  maxBodyChars?: number
}

export function renderCompactToolCall(input: CompactToolCallInput): string
```

Rules:
- Generic renderer, no tool-specific branching.
- Attributes serialized in stable key order.
- If no body -> self-closing tag.
- If body present -> escaped and optionally truncated to `maxBodyChars` with truncation marker.

### `formatResults` changes

- Remove `<results>` wrapper entirely.
- Preserve existing observed-result behavior and oversize guidance.
- Preserve image passthrough exactly.

### `timeline.observation` rendering rules

- Observation text parts render inline at their chronological position with no wrapper tag.
- Observation image parts pass through as image `ContentPart` entries.
- Observations render as standalone chronological items and never create or join agent blocks.

### Full multimodal call chain

```text
memory.transformMessage(context_inbox)
 -> formatContextInbox()
   -> formatResults(results)             // returns text + image parts
   -> renderTimeline(timeline)           // includes standalone observation text/image rendering
   -> appendAttentionTextIfAny()
 => final ContentPart[] (text/image interleaved, no dropped images)
```

---

## Subagent Activity Enrichment

### Types

```ts
export type ActivityAtom =
  | { kind: 'thought'; timestamp: number; text: string }
  | {
      kind: 'tool_call'
      timestamp: number
      toolCallId: string
      tagName: string
      attributes: Record<string, string>
      body?: string
      status: 'success' | 'error' | 'interrupted'
      exitCode?: number
      error?: string
    }
  | {
      kind: 'message'
      timestamp: number
      direction: 'to_lead' | 'from_user' | 'from_lead'
      text: string
    }

export interface ActivityTurnEntry {
  forkId: string
  parentForkId: string | null
  agentId: string
  role: string
  turnId: string
  timestamp: number
  atoms: readonly ActivityAtom[]
}
```

### Data sources by atom kind

- `thought`: subagent `thinking_*` stream (canonical-turn buffers).
- `tool_call`: `turn_completed.toolCalls` + canonical-turn call metadata (`tagName`, `input`, query/body details) + execution status.
- `message`: outbound/inbound message streams classified by direction.

### Signal payload contract

`SubagentActivity.unseenActivityAvailable` payload:

```ts
{
  parentForkId: string | null
  entries: readonly ActivityTurnEntry[]
}
```

Unseen tracking remains cursor-based per parent fork.

### ActivityAtom -> AgentAtom mapping contract

`ActivityAtom` maps into `TimelineEntry.agent_block.atoms` (`AgentAtom`) field-by-field for activity-origin atoms:

- `ActivityAtom.thought` -> `AgentAtom.thought`
  - `timestamp` -> `timestamp`
  - `text` -> `text`
- `ActivityAtom.tool_call` -> `AgentAtom.tool_call`
  - `timestamp` -> `timestamp`
  - `toolCallId` -> `toolCallId`
  - `tagName` -> `tagName`
  - `attributes` -> `attributes`
  - `body` -> `body`
  - `status` -> `status` (required)
  - `exitCode` -> `exitCode`
  - `error` -> `error`
- `ActivityAtom.message` -> `AgentAtom.message`
  - `timestamp` -> `timestamp`
  - `direction` -> `direction`
  - `text` -> `text`

Important source boundary:
- `AgentAtom.idle` and `AgentAtom.error` are not produced from `ActivityAtom`.
- They are produced by non-activity paths (primarily `AgentStatus.agentBecameIdle`, plus turn/error pathways) and appended into agent blocks by memory/composer logic.

### Canonical-turn integration

- Subagent activity reads canonical-turn state to recover structured call metadata and ordering.
- Ordering uses canonical order index to preserve deterministic atom sequence.

---

## Replay and Hydration

### How replay works

- Persisted events are replayed through normal projections during hydration.
- Signals are ephemeral but re-emitted when replayed events trigger emitters.
- Memory has no replay-specific branch; correctness depends on deterministic reduction.

### Replay acceptance criteria

1. Same event log produces byte-equivalent `context_inbox` content after replay.
2. Same queue order under replay/live (`timestamp, seq`).
3. Signal-derived entries reconstruct identically.
4. Coalesced file updates remain idempotent.
5. Mention patching remains idempotent and complete for persisted + queued entries.
6. Deferred presence injects exactly once.
7. Immediate append semantics preserved for unexpected errors.
8. Immediate append semantics preserved for observations.
9. Compaction rewrite behavior unchanged.

### Why design is replay-safe

- All derived state is deterministic from event stream + deterministic signal emission.
- No wall-clock sorting key other than captured event timestamps and per-fork monotonic `seq`.
- Queue flushing remains event-driven (`turn_started`).

---

## Legacy Removal

### Complete fate matrix

#### Message/queue artifacts

- Delete: `comms_inbox`, `system_inbox`, `QueuedCommsMessage`, `QueuedSystemMessage`.
- Keep: `session_context`, `assistant_turn`, `compacted`, `fork_context`.
- Add: `context_inbox`, `QueuedEntry`.

#### Prompt formatters/types

- Delete:
  - `formatCommsInbox`
  - `formatSystemInbox`
  - `formatSubagentActivity`
  - `formatAgentsStatus`
  - `formatAgentIdleNotification`
  - `formatSubagentUserKilledNotification`
  - `formatTaskResult`
  - `formatAgentResponse`
  - `formatLeadMessage`
  - `CommsEntry`
  - `SystemEntry`
  - `AgentActivityEntry`
- Keep:
  - `buildAgentContext`
  - `buildConversationSummary`
- Change:
  - `formatResults` wrapper removed.

#### Observables

- Delete `agents-status-observable.ts`.
- Set `leadObservables = []`.

#### Memory helpers

- Delete legacy helpers tied to old types:
  - `patchCommsEntryMentions`
  - `patchCommsCollections`
  - `appendSystemEntries`
- Remove dead state field:
  - `ForkMemoryState.currentTurnToolCalls` (legacy always-empty field) is deleted and not replaced.
- Replace with context-inbox equivalents:
  - `patchTimelineMentions(...)`
  - `appendContextInboxResults(...)`
  - `appendContextInboxTimeline(...)`

#### Rename summary

- Legacy dual inbox → unified `context_inbox`.
- Legacy activity summary entries → `agent_block` + `AgentAtom[]`.
- Legacy attachment type (`CommsAttachment`) → `TimelineAttachment`.

---

## Consumer Migration

### Chat title worker

File: `packages/agent/src/workers/chat-title-worker.ts`

- Replace `msg.type === 'comms_inbox'` extraction with `msg.type === 'context_inbox'`.
- Source title context from:
  - `timeline.user_message` as user lines.
  - selected agent-to-lead message atoms as assistant-context lines.
- Keep existing `assistant_turn` path.

### Package exports

Files:
- `packages/agent/src/prompts/index.ts`
- `packages/agent/src/index.ts`

Required changes:
- Remove legacy formatter/type exports listed above.
- Export new context types and renderers:
  - `ResultEntry`, `TimelineEntry`, `AgentAtom`, `TimelineAttachment`, `PhaseCriteriaPayload`, `QueuedEntry`
  - `formatContextInbox`
  - `renderCompactToolCall`

### Test updates

- Rewrite tests asserting `system_inbox`/`comms_inbox`.
- Remove obsolete observable tests.
- Add new:
  - context stream formatting tests
  - tool-call renderer tests
  - memory queue/flush/replay parity tests
  - subagent activity enrichment tests
  - chat title worker `context_inbox` extraction tests

---

## Behavioral Changes Register

Intentional/accepted changes:

1. **New context output format** (results lane + flat timeline + optional attention; no wrappers).
2. **Global agent status roster removed** (activity/state-transition-only visibility).
3. **NEW `user_to_agent` parent notification** for subfork-targeted user messages.
4. **Unified queue ordering** replaces dual-lane flush ordering (accepted ordering difference for near-simultaneous events).

All other event handling, queue lifecycle, mention patching, coalescing, immediate append, presence injection, and replay behaviors are preserved (via equivalent typed representation where structures changed).

API cleanup note:
- Public API exports of deleted legacy types/functions are removed as part of migration.
- Consumer audit found no in-repo runtime consumers outside the agent package; this is classified as API surface cleanup, not a behavioral change.

---

## Implementation Plan (5 steps)

### Step 1 — Add new prompt primitives

Files:
- `packages/agent/src/prompts/context-types.ts` (new)
- `packages/agent/src/prompts/tool-call-render.ts` (new)
- `packages/agent/src/prompts/context-stream.ts` (new)
- `packages/agent/src/prompts/results.ts` (edit)
- `packages/agent/src/prompts/index.ts` (add new exports)

State after step:
- Compiles.
- Old runtime path still active.
- New renderer tested in isolation.

### Step 2 — Atomic projection cutover (memory + subagent activity)

Files:
- `packages/agent/src/projections/context-composer.ts` (new)
- `packages/agent/src/projections/memory.ts` (major rewrite)
- `packages/agent/src/projections/subagent-activity.ts` (atoms payload)
- `packages/agent/src/projections/canonical-turn.ts` (metadata support)
- related projection tests

Exact changes:
- Switch message union to `context_inbox`.
- Switch queue to `QueuedEntry`.
- Implement full handler mapping table.
- Keep immediate append semantics.
- Implement mention patching over timeline attachments.

State after step:
- No dual inbox runtime remains.
- Core projection behavior preserved under new IR.

### Step 3 — Remove legacy formatters/observables

Files:
- `packages/agent/src/prompts/agents.ts` (remove old formatters/types)
- `packages/agent/src/prompts/index.ts` (remove old exports)
- `packages/agent/src/observables/agents-status-observable.ts` (delete)
- `packages/agent/src/observables/index.ts` (edit)
- `packages/agent/src/agents/lead-shared.ts` (leadObservables empty)

State after step:
- No legacy formatting/roster pathways remain in code.

### Step 4 — Migrate consumers and public exports

Files:
- `packages/agent/src/workers/chat-title-worker.ts`
- `packages/agent/src/index.ts`
- all affected tests and snapshots

State after step:
- All consumers compile against new context types.
- Public API surface contains no removed legacy types.

### Step 5 — Verification gate

Run:
- `cd packages/agent && npx tsc --noEmit`
- Targeted tests for prompts/projections/workers/replay.
- Grep zero checks:

```bash
rg "system_inbox|comms_inbox|QueuedCommsMessage|QueuedSystemMessage" packages/agent/src
rg "formatSystemInbox|formatCommsInbox|formatSubagentActivity|formatAgentsStatus|formatAgentIdleNotification|formatSubagentUserKilledNotification" packages/agent/src
rg "agentsStatusObservable" packages/agent/src
rg "toolsCalled|filesWritten|prose" packages/agent/src/projections
```

State after step:
- Clean migration complete, no legacy residue.

---

## Acceptance Checklist

- [ ] `Message` union contains exactly: `session_context`, `assistant_turn`, `compacted`, `fork_context`, `context_inbox`.
- [ ] Queue contains only `QueuedEntry` with `lane: result|timeline`.
- [ ] All event and signal handlers are mapped per orchestration table.
- [ ] `agent_created` fork initialization preserved (spawn-only), including optional `fork_context` injection and parentOnSpawn lifecycle reminder behavior.
- [ ] Outbound message target-fork routing is preserved.
- [ ] `observations_captured` is preserved via immediate timeline append.
- [ ] `turn_unexpected_error` is preserved via immediate results append.
- [ ] `turn_completed` empty-response anomaly path is preserved (`EMPTY_RESPONSE_ERROR`).
- [ ] Assistant-turn canonical XML fallback behavior is preserved when canonical XML is unavailable/unclean.
- [ ] Mention resolution patches persisted + queued timeline user message attachments.
- [ ] File-update coalescing by path is preserved.
- [ ] Deferred user presence injection is preserved.
- [ ] No wrappers (`<system>`, `<comms>`, `<results>`) in rendered context.
- [ ] Agent block rendering follows minute-boundary split rules.
- [ ] Attention generation is factual and qualification-limited.
- [ ] `formatResults` still preserves multimodal parts.
- [ ] Subagent activity payload is enriched atoms-based and deterministic.
- [ ] Replay acceptance criteria all pass.
- [ ] `agents-status-observable` is deleted and lead wiring removed.
- [ ] Chat title worker migrated to `context_inbox`.
- [ ] Legacy types/functions/exports removed from public barrels.
- [ ] Grep verification returns zero matches for legacy artifacts.
- [ ] Typecheck and targeted test suites pass.
