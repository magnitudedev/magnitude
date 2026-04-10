# Task System Implementation Spec

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Module: Task Type Registry](#3-module-task-type-registry)
4. [Module: Events](#4-module-events)
5. [Module: TaskGraphProjection](#5-module-taskgraphprojection)
6. [Module: Reader Tag + Wiring](#6-module-reader-tag--wiring)
7. [Module: Task Tools](#7-module-task-tools)
8. [Module: Lead Prompt & Policy](#8-module-lead-prompt--policy)
9. [Module: CLI UI Migration](#9-module-cli-ui-migration)
10. [Implementation Phases](#10-implementation-phases)
11. [Open Questions](#11-open-questions)
12. [Inbox Task Tree View](#12-inbox-task-tree-view)
13. [Appendix: Future Enhancements](#13-appendix-future-enhancements)

---

## 1. Overview

Replace the current worker-first task tracking (derived from `fork_activity` display messages) with a first-class, lead-owned task system.

### Core Principles

- **Tasks are the primary work unit.** All work is organized through tasks. Workers cannot be started without a task.
- **Lead owns task state.** The lead explicitly creates, assigns, completes, and cancels tasks. Worker idle/completion does not automatically change task status.
- **Task types constrain behavior.** Each task type defines allowed assignees, strategy guidance, and suggested decomposition. Higher-order types (feature/bug/refactor) require self-assignment; leaf types allow worker assignment.
- **Workers are tied to tasks.** Assigning a worker to a task starts that worker. Workers live and die with their task assignment.
- **Clean replacement.** No backward compatibility with `agent-create`/`agent-kill`. No legacy bridges.

### Key Decisions

- Task types use a modular registry following `defineRole()`/`defineTool()` patterns
- Allowed assignees reference existing `AgentVariant` + spawnable roles — no redundant role system
- Dependencies are deferred to v2 (appendix)
- External terminology: "worker". Internal code: `agentId`/`forkId`
- Skill system is untouched/orthogonal

---

## 2. Architecture

### Data Flow

```
Task Type Registry (packages/agent/src/tasks/)
  ↓ validation + guidance
Task Tools (create-task, update-task, assign-task, cancel-task)
  ↓ publish events
Events (task_created, task_updated, task_assigned, task_completed, task_cancelled)
  ↓ consumed by
TaskGraphProjection (packages/agent/src/projections/task-graph.ts)
  ↓ reads AgentStatusProjection signals for working/pending derivation
  ↓ exposes state via
TaskGraphStateReaderTag (packages/agent/src/tools/task-reader.ts)
  ↓ wired in ExecutionManager.makeForkLayers()
  ↓ exposed to CLI via CodingAgent.expose
CLI UI (use-tasks hook → task-list.tsx tree rendering)
```

### Integration with Existing Agent Lifecycle

- `assign-task` tool calls `ExecutionManager.fork()` to spawn workers — same infrastructure as current `agent-create`
- `ExecutionManager.fork()` publishes `agent_created` event internally — `AgentStatusProjection` and `AgentLifecycle` worker continue functioning unchanged
- `cancel-task` and reassignment publish `agent_killed` events — same kill path as current `agent-kill`
- `TaskGraphProjection` subscribes to `AgentStatusProjection` signals (`agentBecameWorking`, `agentBecameIdle`) for working/pending status derivation — never auto-completes
- `agent-create`/`agent-kill` tools are removed from lead tool surface and catalog

---

## 3. Module: Task Type Registry

### File Structure

```
packages/agent/src/tasks/
  types.ts            — Core interfaces
  define.ts           — defineTaskType() constructor
  registry.ts         — TASK_TYPES record + helpers
  guidance.ts         — Prompt guidance renderer
  validation.ts       — Reusable validation for tools/projection
  index.ts            — Public exports
  definitions/
    feature.ts
    bug.ts
    refactor.ts
    research.ts
    plan.ts
    implement.ts
    review.ts
    other.ts
    index.ts
```

### `types.ts`

```ts
import type { AgentVariant } from '../agents'

export type TaskAssignee = 'self' | AgentVariant

export interface SuggestedChildTask {
  readonly type: string
  readonly title: string
  readonly rationale?: string
}

export interface TaskTypeGuidance {
  readonly strategy: string
  readonly suggestedWorkers: readonly AgentVariant[]
  readonly suggestedChildTasks?: readonly SuggestedChildTask[]
}

export interface TaskTypeDefinition<TId extends string = string> {
  readonly id: TId
  readonly label: string
  readonly description: string
  readonly allowedAssignees: readonly TaskAssignee[]
  readonly guidance: TaskTypeGuidance
}
```

### `define.ts`

```ts
import { getSpawnableVariants, isValidVariant } from '../agents'
import type { TaskTypeDefinition } from './types'

export function defineTaskType<const TId extends string>(
  definition: TaskTypeDefinition<TId>,
): TaskTypeDefinition<TId> {
  if (definition.allowedAssignees.length === 0) {
    throw new Error(`Task type "${definition.id}" must declare at least one allowed assignee.`)
  }

  const spawnableVariants = new Set(getSpawnableVariants())

  for (const assignee of definition.allowedAssignees) {
    if (assignee === 'self') continue

    if (!isValidVariant(assignee)) {
      throw new Error(
        `Task type "${definition.id}" has invalid assignee "${assignee}". ` +
          `Expected "self" or a valid AgentVariant.`,
      )
    }

    if (!spawnableVariants.has(assignee)) {
      throw new Error(
        `Task type "${definition.id}" references non-spawnable assignee "${assignee}". ` +
          `Only spawnable AgentVariants may be worker assignees.`,
      )
    }
  }

  return definition
}
```

### `definitions/feature.ts`

```ts
import { defineTaskType } from '../define'

export const featureTaskType = defineTaskType({
  id: 'feature',
  label: 'Feature',
  description: 'Deliver a user-facing capability through deliberate decomposition and orchestration.',
  allowedAssignees: ['self'],
  guidance: {
    strategy:
      'Own orchestration directly. Decompose into research/plan/implement/review tasks; ' +
      'verify completion criteria and user intent before marking complete.',
    suggestedWorkers: ['explorer', 'planner', 'builder', 'reviewer'],
    suggestedChildTasks: [
      { type: 'research', title: 'Research existing behavior, constraints, and integration points', rationale: 'Ground implementation in evidence before planning changes.' },
      { type: 'plan', title: 'Plan implementation strategy and user-facing tradeoffs', rationale: 'Lock down approach and sequencing before coding.' },
      { type: 'implement', title: 'Implement the approved feature changes', rationale: 'Execute scoped code edits against the plan.' },
      { type: 'review', title: 'Review integrated result for correctness and quality', rationale: 'Catch correctness, regressions, and requirement misses before completion.' },
    ],
  },
} as const)
```

### `definitions/bug.ts`

```ts
import { defineTaskType } from '../define'

export const bugTaskType = defineTaskType({
  id: 'bug',
  label: 'Bug',
  description: 'Resolve a defect with evidence-driven diagnosis and validation.',
  allowedAssignees: ['self'],
  guidance: {
    strategy:
      'Run evidence-first root cause analysis. Reproduce clearly, isolate the failing behavior, ' +
      'apply minimal targeted fixes, and verify with red/green validation before completion.',
    suggestedWorkers: ['debugger', 'explorer', 'builder', 'reviewer'],
    suggestedChildTasks: [
      { type: 'research', title: 'Reproduce the bug and gather concrete failure evidence', rationale: 'Avoid speculative fixes by anchoring to observable behavior.' },
      { type: 'plan', title: 'Plan root-cause fix scope and validation strategy', rationale: 'Ensure fix addresses cause, not only symptoms.' },
      { type: 'implement', title: 'Implement and test the bug fix', rationale: 'Apply targeted change with verification.' },
      { type: 'review', title: 'Review fix robustness and regression risk', rationale: 'Confirm no collateral breakage before completion.' },
    ],
  },
} as const)
```

### `definitions/refactor.ts`

```ts
import { defineTaskType } from '../define'

export const refactorTaskType = defineTaskType({
  id: 'refactor',
  label: 'Refactor',
  description: 'Improve code structure while preserving behavior.',
  allowedAssignees: ['self'],
  guidance: {
    strategy:
      'Treat behavior preservation as non-negotiable. Use incremental, test-backed structural changes, ' +
      'keeping the system green throughout and verifying no external behavior drift.',
    suggestedWorkers: ['explorer', 'builder', 'reviewer'],
    suggestedChildTasks: [
      { type: 'research', title: 'Map current structure, constraints, and safety checks', rationale: 'Identify what must stay behaviorally stable.' },
      { type: 'plan', title: 'Plan incremental refactor slices with verification checkpoints', rationale: 'Reduce risk with small reversible steps.' },
      { type: 'implement', title: 'Execute refactor slices while keeping tests and behavior green', rationale: 'Maintain confidence continuously, not only at the end.' },
      { type: 'review', title: 'Review architectural quality and behavioral parity', rationale: 'Confirm improved structure without regressions.' },
    ],
  },
} as const)
```

### `definitions/research.ts`

```ts
import { defineTaskType } from '../define'

export const researchTaskType = defineTaskType({
  id: 'research',
  label: 'Research',
  description: 'Investigate systems/code to reduce uncertainty and inform decisions.',
  allowedAssignees: ['self', 'explorer'],
  guidance: {
    strategy:
      'Prioritize high-signal evidence gathering. Report concrete findings, unknowns, and ' +
      'implications for next steps. Prefer source-backed conclusions over assumptions.',
    suggestedWorkers: ['explorer'],
  },
} as const)
```

### `definitions/plan.ts`

```ts
import { defineTaskType } from '../define'

export const planTaskType = defineTaskType({
  id: 'plan',
  label: 'Plan',
  description: 'Produce an executable implementation plan aligned with requirements.',
  allowedAssignees: ['self', 'planner'],
  guidance: {
    strategy:
      'Translate requirements into explicit implementation units, constraints, and validation steps. ' +
      'Surface tradeoffs and open questions early; seek user/lead confirmation when needed.',
    suggestedWorkers: ['planner'],
  },
} as const)
```

### `definitions/implement.ts`

```ts
import { defineTaskType } from '../define'

export const implementTaskType = defineTaskType({
  id: 'implement',
  label: 'Implement',
  description: 'Execute concrete code changes against a defined objective.',
  allowedAssignees: ['self', 'builder'],
  guidance: {
    strategy:
      'Apply scoped edits that fit existing code patterns, then validate with relevant tests/checks. ' +
      'Report exactly what changed and what was verified.',
    suggestedWorkers: ['builder'],
  },
} as const)
```

### `definitions/review.ts`

```ts
import { defineTaskType } from '../define'

export const reviewTaskType = defineTaskType({
  id: 'review',
  label: 'Review',
  description: 'Critically evaluate work for correctness, quality, and requirement fit.',
  allowedAssignees: ['self', 'reviewer'],
  guidance: {
    strategy:
      'Review with adversarial rigor: requirement coverage, correctness, edge cases, regressions, ' +
      'and code quality. Provide actionable findings and iterate until resolved.',
    suggestedWorkers: ['reviewer'],
  },
} as const)
```

### `definitions/other.ts`

```ts
import { defineTaskType } from '../define'

export const otherTaskType = defineTaskType({
  id: 'other',
  label: 'Other',
  description: 'Catch-all for work that does not cleanly map to predefined task types.',
  allowedAssignees: ['self', 'explorer', 'planner', 'builder', 'reviewer', 'debugger', 'browser'],
  guidance: {
    strategy:
      'Use this only when no specific task type fits. State intent clearly, define explicit success criteria, ' +
      'and choose the most appropriate assignee for the actual work.',
    suggestedWorkers: ['explorer', 'planner', 'builder', 'reviewer', 'debugger', 'browser'],
  },
} as const)
```

### `definitions/index.ts`

```ts
export { featureTaskType } from './feature'
export { bugTaskType } from './bug'
export { refactorTaskType } from './refactor'
export { researchTaskType } from './research'
export { planTaskType } from './plan'
export { implementTaskType } from './implement'
export { reviewTaskType } from './review'
export { otherTaskType } from './other'
```

### `registry.ts`

```ts
import type { TaskAssignee, TaskTypeDefinition, TaskTypeGuidance } from './types'
import {
  bugTaskType,
  featureTaskType,
  implementTaskType,
  otherTaskType,
  planTaskType,
  refactorTaskType,
  researchTaskType,
  reviewTaskType,
} from './definitions'

export type TaskTypeId = 'feature' | 'bug' | 'refactor' | 'research' | 'plan' | 'implement' | 'review' | 'other'

export const TASK_TYPES: Record<TaskTypeId, TaskTypeDefinition<TaskTypeId>> = {
  feature: featureTaskType,
  bug: bugTaskType,
  refactor: refactorTaskType,
  research: researchTaskType,
  plan: planTaskType,
  implement: implementTaskType,
  review: reviewTaskType,
  other: otherTaskType,
}

export function isValidTaskType(value: string): value is TaskTypeId {
  return Object.hasOwn(TASK_TYPES, value)
}

export function getTaskTypeDefinition(taskType: TaskTypeId): TaskTypeDefinition<TaskTypeId> {
  return TASK_TYPES[taskType]
}

export function listTaskTypeDefinitions(): readonly TaskTypeDefinition<TaskTypeId>[] {
  return Object.values(TASK_TYPES)
}

export function isTaskAssigneeAllowed(taskType: TaskTypeId, assignee: TaskAssignee): boolean {
  return TASK_TYPES[taskType].allowedAssignees.includes(assignee)
}

export function getTaskTypeGuidance(taskType: TaskTypeId): TaskTypeGuidance {
  return TASK_TYPES[taskType].guidance
}
```

### `guidance.ts`

Two functions: a lightweight reference table for the system prompt, and a task creation reminder formatter for the inbox system.

```ts
import { listTaskTypeDefinitions, type TaskTypeId } from './registry'

/**
 * Lightweight reference table for system prompt — types + allowed assignees only.
 * Strategy/child structure guidance comes via inbox hooks on task creation.
 */
export function renderTaskTypeReferenceTable(): string {
  const lines: string[] = []
  lines.push('<task_types>')
  for (const def of listTaskTypeDefinitions()) {
    lines.push(`  <type id="${def.id}" label="${def.label}" assignees="${def.allowedAssignees.join(', ')}" />`)
  }
  lines.push('</task_types>')
  return lines.join('\n')
}

/**
 * Task creation reminder formatter — called by inbox system when task_type_hook
 * timeline entries are rendered. Receives consolidated taskIds grouped by type.
 * Same pattern as LifecycleReminderFormatter(agentIds).
 */
export function formatTaskTypeReminder(taskIds: readonly string[], taskType: TaskTypeId): string {
  const def = listTaskTypeDefinitions().find(d => d.id === taskType)
  if (!def) return `Tasks ${taskIds.join(', ')} created (unknown type: ${taskType}).`

  const idList = taskIds.length === 1 ? `Task ${taskIds[0]}` : `Tasks ${taskIds.join(', ')}`
  const lines: string[] = []
  lines.push(`${idList} (type: ${def.label}):`)
  lines.push(`Strategy: ${def.guidance.strategy}`)
  if (def.guidance.suggestedWorkers.length > 0) {
    lines.push(`Suggested workers: ${def.guidance.suggestedWorkers.join(', ')}`)
  }
  if (def.guidance.suggestedChildTasks && def.guidance.suggestedChildTasks.length > 0) {
    lines.push('Suggested child tasks:')
    for (const child of def.guidance.suggestedChildTasks) {
      lines.push(`- ${child.type}: ${child.title}`)
    }
  }
  return lines.join('\n')
}
```
```ts
export * from './types'
export * from './define'
export * from './registry'
export * from './guidance'
export * from './validation'
export * from './definitions'
```

---

## 4. Module: Events

Add to `packages/agent/src/events.ts`:

### Imports

```ts
import type { TaskTypeId, TaskAssignee } from './tasks'
```

### Event Interfaces

```ts
// Task Events

export interface TaskCreated {
  readonly type: 'task_created'
  readonly forkId: string | null
  readonly taskId: string
  readonly title: string
  readonly taskType: TaskTypeId
  readonly parentId: string | null
  readonly timestamp: number
}

export interface TaskUpdated {
  readonly type: 'task_updated'
  readonly forkId: string | null
  readonly taskId: string
  readonly patch: {
    readonly title?: string
    readonly parentId?: string | null
  }
  readonly timestamp: number
}

export interface TaskAssigned {
  readonly type: 'task_assigned'
  readonly forkId: string | null
  readonly taskId: string
  readonly assignee: TaskAssignee
  readonly workerRole?: string
  readonly message: string
  readonly workerInfo?: {
    readonly agentId: string
    readonly forkId: string
    readonly role: string
  }
  readonly replacedWorker?: {
    readonly agentId: string
    readonly forkId: string
  }
  readonly timestamp: number
}

export interface TaskCompleted {
  readonly type: 'task_completed'
  readonly forkId: string | null
  readonly taskId: string
  readonly timestamp: number
}

export interface TaskCancelled {
  readonly type: 'task_cancelled'
  readonly forkId: string | null
  readonly taskId: string
  readonly cancelledSubtree: readonly string[]
  readonly killedWorkers: readonly {
    readonly agentId: string
    readonly forkId: string
  }[]
  readonly timestamp: number
}
```

### AppEvent Union Addition

```ts
export type AppEvent =
  // ... existing variants ...
  | TaskCreated
  | TaskUpdated
  | TaskAssigned
  | TaskCompleted
  | TaskCancelled
```

---

## 5. Module: TaskGraphProjection

File: `packages/agent/src/projections/task-graph.ts`

### State Types

```ts
export type TaskStatus = 'pending' | 'working' | 'completed'

export interface TaskRecord {
  readonly id: string
  readonly title: string
  readonly taskType: TaskTypeId
  readonly parentId: string | null
  readonly childIds: readonly string[]
  readonly assignee: TaskAssignee | null
  readonly worker: {
    readonly agentId: string
    readonly forkId: string
    readonly role: string
    readonly message: string
  } | null
  readonly status: TaskStatus
  readonly createdAt: number
  readonly updatedAt: number
  readonly completedAt: number | null
}

export interface TaskGraphState {
  readonly tasks: ReadonlyMap<string, TaskRecord>
  readonly rootTaskIds: readonly string[]
}
```

### Projection Implementation

Uses `Projection.define` (singleton — tasks are global, not per-fork).

**Event handlers:**
- `task_created`: Validate unique ID, parent exists. Insert pending task. Add to parent's childIds or rootTaskIds. If parent was completed, transition parent to pending.
- `task_updated`: Validate task exists. Apply title rename and/or reparent (with cycle detection — cannot parent under own descendant).
- `task_assigned`: Validate assignee allowed via `isTaskAssigneeAllowed`. Update assignee/worker linkage. Derive status (self=pending, worker=working).
- `task_completed`: Validate all children completed. Set status=completed, completedAt.
- `task_cancelled`: Remove entire subtree from state. Clean up parent's childIds.

**Signal handlers:**
- Subscribe to `AgentStatusProjection.signals.agentBecameWorking` and `agentBecameIdle`
- Map agent/fork back to linked task via worker field
- Recalculate working/pending status — never auto-complete

**Reads:** `[AgentStatusProjection]`

**Signals exposed:** `taskCreated`, `taskCompleted`, `taskCancelled`, `taskStatusChanged`

**Helper functions:**
- `collectSubtreeTaskIds(state, rootId)` — DFS subtree collection
- `canCompleteTask(state, taskId)` — all children completed check
- `patchTask(state, taskId, updater)` — immutable task update
- `reparentTask(state, taskId, newParentId, timestamp)` — with cycle detection

See `$M/plans/events-projection.md` for complete TypeScript implementation.

---

## 6. Module: Reader Tag + Wiring

### `packages/agent/src/tools/task-reader.ts`

```ts
import { Context, Effect } from 'effect'
import type { TaskAssignee } from '../tasks'
import type { TaskGraphState, TaskRecord } from '../projections/task-graph'

export interface TaskGraphStateReader {
  readonly getTask: (id: string) => Effect.Effect<TaskRecord | undefined>
  readonly getState: () => Effect.Effect<TaskGraphState>
  readonly getChildren: (id: string) => Effect.Effect<readonly TaskRecord[]>
  readonly canComplete: (id: string) => Effect.Effect<boolean>
  readonly canAssign: (id: string, assignee: TaskAssignee) => Effect.Effect<boolean>
  readonly getSubtree: (id: string) => Effect.Effect<readonly TaskRecord[]>
}

export class TaskGraphStateReaderTag extends Context.Tag('TaskGraphStateReader')<
  TaskGraphStateReaderTag,
  TaskGraphStateReader
>() {}
```

### ExecutionManager Wiring

In `packages/agent/src/execution/execution-manager.ts`, in `makeForkLayers()`:

```ts
const taskGraphReaderLayer = Layer.succeed(TaskGraphStateReaderTag, {
  getTask: (id) => Effect.map(taskGraphProjection.get, (s) => s.tasks.get(id)),
  getState: () => taskGraphProjection.get,
  getChildren: (id) => Effect.map(taskGraphProjection.get, (s) => getChildRecords(s, id)),
  canComplete: (id) => Effect.map(taskGraphProjection.get, (s) => canCompleteRecord(s, id)),
  canAssign: (id, assignee) => Effect.map(taskGraphProjection.get, (s) => canAssignRecord(s, id, assignee)),
  getSubtree: (id) => Effect.map(taskGraphProjection.get, (s) => collectSubtreeRecords(s, id)),
})
```

Merge into `Layer.mergeAll(...)` alongside existing reader layers.

### CodingAgent Registration

In `packages/agent/src/coding-agent.ts`:

- Add `TaskGraphProjection` to projections list **after** `AgentStatusProjection`
- Expose in `expose.state`: `taskGraph: TaskGraphProjection`
- Expose signals: `taskCreated`, `taskCompleted`, `taskCancelled`, `taskStatusChanged`

---

## 7. Module: Task Tools

File: `packages/agent/src/tools/task-tools.ts`

### 7.1 create-task

**XML:**
```
<create-task id="t-research-auth" type="research" parent="t-feature-auth">Research auth token refresh flow</create-task>
```

**Input:** `{ taskId, type, parent?, title }` → **Output:** `{ taskId }`

**Binding:** `id`/`type`/`parent` as attrs, body as `title`

**Execute:** Validate type via `isValidTaskType`, validate parent exists via `TaskGraphStateReaderTag`, publish `task_created`.

### 7.2 update-task

**XML:**
```
<update-task id="t-plan-api" parent="t-feature-v2">New title</update-task>
<update-task id="t-plan-api" complete="true" />
```

**Input:** `{ taskId, parent?, complete?, title? }` → **Output:** `{ taskId }`

**Binding:** `id`/`parent`/`complete` as attrs, body as `title`

**Execute:** Require at least one mutation. If `complete=true`, validate via `canComplete` then publish `task_completed`. Otherwise publish `task_updated`.

### 7.3 assign-task

**XML:**
```
<assign-task id="t-research-auth" assignee="explorer">Investigate the token refresh flow</assign-task>
<assign-task id="t-feature-auth" assignee="self" />
```

**Input:** `{ taskId, assignee, message? }` → **Output:** `{ taskId, agentId?, forkId? }`

**Binding:** `id`/`assignee` as attrs, body as `message`

**Execute:**
1. Read task via `TaskGraphStateReaderTag`
2. Validate assignee via `isTaskAssigneeAllowed`
3. If existing worker: publish `agent_killed`
4. If `assignee="self"`: publish `task_assigned` with lead assignee
5. If worker role: require message body, build context (same pattern as current `agent-create` — `ConversationStateReaderTag`, `buildConversationSummary`, `buildAgentContext`), call `ExecutionManager.fork()`, publish `task_assigned` with worker linkage
6. Worker agentId generated as `${role}-${taskId}`

### 7.4 cancel-task

**XML:**
```
<cancel-task id="t-obsolete" />
```

**Input:** `{ taskId }` → **Output:** `{ taskId, cancelledCount, workersKilled }`

**Binding:** `id` as attr, no body

**Execute:** Collect subtree, kill all linked workers (`agent_killed` per worker), publish `task_cancelled`.

### State Models

One model per tool in `packages/agent/src/models/`:
- `create-task.ts` — tracks taskId, type, title, phase
- `update-task.ts` — tracks taskId, parent, complete, title, phase
- `assign-task.ts` — tracks taskId, assignee, agentId, forkId, phase
- `cancel-task.ts` — tracks taskId, cancelledCount, workersKilled, phase

All follow `defineStateModel` pattern matching existing `agentCreateModel`/`agentKillModel`.

### Catalog Registration

In `packages/agent/src/catalog.ts`:

**Add:**
```ts
createTask: { tool: createTaskTool, binding: createTaskXmlBinding, state: createTaskModel, display: false },
updateTask: { tool: updateTaskTool, binding: updateTaskXmlBinding, state: updateTaskModel, display: false },
assignTask: { tool: assignTaskTool, binding: assignTaskXmlBinding, state: assignTaskModel, display: false },
cancelTask: { tool: cancelTaskTool, binding: cancelTaskXmlBinding, state: cancelTaskModel, display: false },
```

**Remove:** `agentCreate`, `agentKill` entries.

**Delete files:** `tools/agent-tools.ts`, `models/agent-create.ts`, `models/agent-kill.ts`

See `$M/plans/task-tools.md` for complete TypeScript implementation of all tools and bindings.

---

## 8. Module: Lead Prompt & Policy

### 8.1 `lead.txt` Changes

Replace subagent-oriented workflow sections with task-first equivalents:

**Replace `## Subagent use` with:**

> ## Task-first execution
>
> You manage work through tasks. Do not start workers directly without a task.
>
> Core discipline:
> 1. Create tasks as soon as work becomes clear.
> 2. Keep tasks organized (use parent/child structure where helpful).
> 3. Assign tasks deliberately:
>    - `assignee="self"` means you (lead) own orchestration work for that task.
>    - `assignee="<worker-role>"` starts a worker on that task.
> 4. Treat worker output as input to your decision-making, not completion proof.
> 5. Mark tasks complete only when you have verified they are actually done.

**Replace `## Subagent review` with:**

> ## Worker review discipline
>
> When a worker returns, never assume the task is complete. You own task state and quality bar.
> Review worker output against user intent. Iterate until sufficient. Only mark complete when you judge it complete.

**Replace `## Subagent chaining` with:**

> ## Task decomposition and worker chaining
>
> Use tasks to decompose and sequence work. Compose workers through task handoffs.
> Typical patterns: research (explorer) → plan (planner) → implement (builder) → review (reviewer).

**Replace `## Available subagents` with `## Available worker roles`** (same content, "subagent" → "worker")

**Add before `## Workspace`:**

> ## Task state ownership
>
> - Pending: task exists, no active worker.
> - Working: task has an active worker assignment.
> - Completed: explicitly marked complete by you.
>
> Rules: Assigning starts the worker. Reassigning replaces the worker. Parent tasks require child completion. Completed means archived confidence.

### 8.2 `lead-tooling.txt` Addition

Insert before `{{TOOL_DOCS}}`:

> ## Task tool usage discipline
>
> - `create-task`: define work units. `update-task`: rename/reparent/complete. `assign-task`: assign + start. `cancel-task`: remove branches.
> - Do not dispatch workers without a task. Keep task state accurate. Treat worker completion as review signal, not done-state.

### 8.3 `lead-shared.ts` Updates

**leadTools:**
```ts
export const leadTools = catalog.pick(
  'fileRead', 'fileWrite', 'fileEdit', 'fileTree', 'fileSearch', 'fileView',
  'shell', 'webSearch', 'webFetch',
  'createTask', 'updateTask', 'assignTask', 'cancelTask',
  'skill', 'phaseSubmit',
)
```

**leadTurnPolicy** — replace yield triggers:
```ts
const yielders = ['assignTask', 'cancelTask']
```

### 8.4 Context Injection (Two Layers)

**Layer 1: System prompt — lightweight type reference table (always present)**

In `packages/agent/src/prompts/session-context.ts`:

```ts
import { renderTaskTypeReferenceTable } from '../tasks'
// In buildSessionContextContent():
content += '\n' + renderTaskTypeReferenceTable()
```

This injects a compact reference of task types + allowed assignees so the lead always knows what's available. No strategy text — just the menu.

**Layer 2: Inbox hook — strategy guidance on task creation (on demand)**

Uses the same pattern as `lifecycle_hook` timeline entries. When tasks are created, strategy guidance is delivered through the inbox `<reminders>` block, consolidated by type.

In `packages/agent/src/inbox/types.ts`, add a new `TimelineEntry` variant:

```ts
| (Timestamped<'task_type_hook'> & { readonly taskId: string; readonly taskType: string; readonly title: string })
```

In `packages/agent/src/projections/memory.ts`:
- Handle `task_created` events in event handlers
- Enqueue `task_type_hook` timeline entry into the lead fork timeline

In `packages/agent/src/inbox/render.ts` (`formatInbox`):
- Extract `task_type_hook` entries alongside `lifecycle_hook` entries
- Group by `taskType`, consolidate taskIds (same dedup pattern as lifecycle hooks)
- Call `formatTaskTypeReminder(taskIds, taskType)` from `tasks/guidance.ts`
- Append to `<reminders>` block

Example: if the lead creates 3 research tasks in one turn, the reminder is consolidated:
> Tasks t-research-1, t-research-2, t-research-3 (type: Research):
> Strategy: Prioritize high-signal evidence gathering. Report concrete findings...
> Suggested workers: explorer

---

## 9. Module: CLI UI Migration

### 9.1 `TaskListItem` Type

In `cli/src/components/chat/types.ts`, replace `TaskItem` with:

```ts
export type TaskListItem = {
  taskId: string
  title: string
  type: string
  status: 'pending' | 'working' | 'completed'
  depth: number
  parentId: string | null
  createdAt: number
  updatedAt: number
  completedAt: number | null
  assignee: { kind: 'lead' } | { kind: 'worker'; workerType?: string; agentId: string }
  workerForkId: string | null
}
```

### 9.2 Tree Utility

New file: `cli/src/utils/task-tree.ts`

`flattenTaskTree(state: TaskGraphState): TaskListItem[]` — DFS traversal producing depth-annotated flat list. Roots and siblings sorted by `createdAt` ascending.

### 9.3 Hook Rewrite

`cli/src/hooks/use-tasks.ts` — Subscribe to `client.state.taskGraph` instead of `fork_activity`. Use `flattenTaskTree`. Still subscribe to worker fork display for pending user-message active hint.

### 9.4 Tree Rendering

`cli/src/components/chat/task-list.tsx`:
- Indentation: `'  '.repeat(depth)` + `└─ ` for non-root
- Status glyphs: pending `○`, working `◉` (pulsing blue), completed `✓` (green)
- Type badge: `[implement]` before title
- Assignee: `lead` (plain) or clickable worker label → `pushForkOverlay(workerForkId)`
- Keep panel-level collapse/expand

---

## 10. Implementation Phases

### Phase 1: Task Type Registry + Events + Projection + Reader Tag

**Files:** `packages/agent/src/tasks/**`, `events.ts`, `projections/task-graph.ts`, `tools/task-reader.ts`, `execution-manager.ts`, `coding-agent.ts`

**Deliverables:** Task types defined and validated, events in AppEvent union, TaskGraphProjection with all rules, reader tag wired, state exposed to client.

**Functional outcome:** Tasks can be created/updated/completed/cancelled via direct event tests.

### Worker Idle → Task-Aware Reminder (part of Phase 1)

Uses the same inbox hook pattern as task creation guidance.

In `packages/agent/src/projections/memory.ts`:
- Add `TaskGraphProjection` to `reads`
- In the `agentBecameIdle` signal handler, after the existing role-based lifecycle hook, look up the linked task and enqueue a `task_idle_hook` timeline entry

In `packages/agent/src/inbox/types.ts`, add:
```ts
| (Timestamped<'task_idle_hook'> & { readonly taskId: string; readonly taskType: string; readonly title: string; readonly agentId: string })
```

In `packages/agent/src/inbox/render.ts`:
- Extract `task_idle_hook` entries, group by taskType, consolidate
- Render reminder: "Worker X for task Y has finished. Review output and either send feedback to iterate or mark complete."
- Append to `<reminders>` block alongside lifecycle and task creation reminders

This reuses the existing inbox/timeline infrastructure with zero new systems.

### Phase 2: Task Tools + Catalog + Remove Agent Tools

**Files:** `tools/task-tools.ts`, `models/create-task.ts`, `models/update-task.ts`, `models/assign-task.ts`, `models/cancel-task.ts`, `catalog.ts`, `models/index.ts`

**Deliverables:** All 4 tools implemented with XML bindings and state models. Catalog updated. `agent-create`/`agent-kill` removed.

**Functional outcome:** Lead manages tasks and workers exclusively through task tools.

**Depends on:** Phase 1

### Phase 3: Lead Prompt & Policy

**Files:** `agents/prompts/lead.txt`, `agents/prompts/lead-tooling.txt`, `agents/lead-shared.ts`, `prompts/session-context.ts`

**Deliverables:** Prompt rewritten for task-first workflow. Turn policy updated. Guidance injected from registry.

**Functional outcome:** Lead operates task-first with mechanically enforced task type constraints.

**Depends on:** Phase 2

### Phase 4: CLI UI Migration

**Files:** `cli/src/components/chat/types.ts`, `cli/src/utils/task-tree.ts`, `cli/src/hooks/use-tasks.ts`, `cli/src/components/chat/task-list.tsx`, `cli/src/components/chat/task-list.test.tsx`

**Deliverables:** Tree rendering, hook rewrite, updated tests.

**Functional outcome:** User sees hierarchical task tree with status, type, and assignee.

**Depends on:** Phase 1 (for state exposure)

---

## 11. Open Questions

1. **Per-node collapse/expand** — Current plan uses panel-level collapse. Per-node expand/collapse is a follow-up.
2. **Task type display format** — Raw token `[implement]` or title-cased `[Implement]`? Color-coded?
3. **Timer display** — Should tree UI show elapsed time? Based on `createdAt`/`assignedAt` or worker-derived timing?
4. **`other` type future-proofing** — Currently lists all spawnable variants explicitly. Auto-derive from registry if new variants are added?

---

## 12. Inbox Task Tree View

### Purpose

Show the lead relevant parts of the task tree at appropriate moments — scoped to affected subtrees, deduplicated, token-efficient.

### When to Show

1. **After any turn where task tools modified the graph** (create-task, update-task, assign-task, cancel-task)
2. **When a worker goes idle** and is linked to a task

### What to Show

- For each affected taskId, walk up to the root ancestor
- Deduplicate roots — if A1 and A1i both changed, they share root A, render A's tree once
- Render each unique root's full subtree as an indented text tree

### Rendering Format (token-efficient, no unicode)

```
<task_tree>
[pending] feature: Implement auth refresh (t-feature-auth, lead)
  [working] research: Research token flow (t-research, explorer-t-research)
  [done] plan: Plan implementation (t-plan)
  [pending] implement: Build refresh logic (t-impl)
</task_tree>
```

Format per line: `{indent}[{status}] {type}: {title} ({taskId}{, assignee if assigned})`
- Status tokens: `pending`, `working`, `done`
- Indent: 2 spaces per depth level
- Assignee shown only if assigned (agentId for workers, `lead` for self)

### Implementation

**New timeline entry** in `packages/agent/src/inbox/types.ts`:

```ts
| (Timestamped<'task_tree_view'> & { readonly rootTaskIds: readonly string[] })
```

**MemoryProjection** (`packages/agent/src/projections/memory.ts`):
- Add `TaskGraphProjection` to `reads` (already added for worker idle hook)
- In event handlers for `task_created`, `task_updated`, `task_assigned`, `task_completed`, `task_cancelled`:
  - Extract affected taskId from event
  - Walk up to root ancestor using TaskGraphProjection state
  - Enqueue `task_tree_view` timeline entry with the root taskId
- In `agentBecameIdle` signal handler (already has task lookup):
  - If linked task found, also enqueue `task_tree_view` with that task's root ancestor
- Coalescing: use `coalesceKey: 'task_tree_view'` so multiple task events in the same inbox delivery merge into one entry with combined rootTaskIds

**formatInbox** (`packages/agent/src/inbox/render.ts`):
- Extract `task_tree_view` entries, merge all rootTaskIds, deduplicate
- Read TaskGraphProjection state to render each root's subtree
- Helper function `renderTaskSubtree(state: TaskGraphState, rootId: string, depth: number): string[]`:
  - Renders the root and all descendants recursively with indentation
  - Uses `[pending]`/`[working]`/`[done]` status tokens
- Output wrapped in `<task_tree>...</task_tree>` tags
- Placed after tool results, before other timeline entries

### Scoping Rules

- Only show trees that contain at least one affected task
- Never show the full graph — only affected root subtrees
- If the same root appears multiple times (from multiple affected tasks), render once
- Worker idle events show the tree containing the worker's linked task

---

## 13. Appendix: Future Enhancements

- **Task dependencies:** Add `deps` attr (comma-separated IDs) on `create-task`/`update-task`. Add dependency-aware assignment gating and cycle detection in projection.
- **Per-node tree collapse:** Expand/collapse individual parent nodes in CLI UI.
- **Task filtering/search:** Filter task tree by status, type, or assignee.
