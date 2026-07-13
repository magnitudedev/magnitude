import { Projection, type ReadFn, type ForkedState } from '@magnitudedev/event-core'
import { outcomeWillChainContinue } from '../events'
import type { AppEvent } from '../events'
import { Option, Schema } from 'effect'
import {
  AgentLifecycleProjection,
  getAgentByForkId,
  type AgentInfo,
  type AgentLifecycleState,
} from './agent-lifecycle'
import {
  TaskGraphProjection,
  type TaskGraphState,
  type TaskRecord,
  TaskStatusSchema,
} from './task-graph'
import { HarnessStateProjection, getToolHandlesRecord, type HarnessTurnState } from './harness-state'
import type { ToolState } from '../models'
import type { ToolHandleFromSchema } from '../models/tool-handle-schema'

export const WorkerStateSchema = Schema.Union(
  Schema.Struct({ status: Schema.Literal('unassigned') }),
  Schema.Struct({
    status: Schema.Literal('spawning'),
    toolCallId: Schema.String,
    role: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  }),
  Schema.Struct({
    status: Schema.Literal('working'),
    forkId: Schema.String,
    activeSince: Schema.Number,
    accumulatedMs: Schema.Number,
    resumeCount: Schema.Number,
  }),
  Schema.Struct({
    status: Schema.Literal('idle'),
    forkId: Schema.String,
    accumulatedMs: Schema.Number,
    completedAt: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
    resumeCount: Schema.Number,
  }),
  Schema.Struct({
    status: Schema.Literal('killing'),
    forkId: Schema.String,
    toolCallId: Schema.String,
  }),
)
export type WorkerState = typeof WorkerStateSchema.Type

export const WorkerActivitySchema = Schema.Struct({
  forkId: Schema.String,
  activeSince: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
  accumulatedMs: Schema.Number,
  lastStintMs: Schema.Number,
  completedAt: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
  resumeCount: Schema.Number,
})
export type WorkerActivity = typeof WorkerActivitySchema.Type

export const TaskAssignmentAssigneeSchema = Schema.Union(
  Schema.Struct({ kind: Schema.Literal('none') }),
  Schema.Struct({ kind: Schema.Literal('user') }),
  Schema.Struct({
    kind: Schema.Literal('worker'),
    role: Schema.String,
    agentId: Schema.String,
    forkId: Schema.String,
  }),
)
export type TaskAssignmentAssignee = typeof TaskAssignmentAssigneeSchema.Type

export const TaskAssignmentRowSchema = Schema.Struct({
  taskId: Schema.String,
  title: Schema.String,
  status: TaskStatusSchema,
  parentId: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  depth: Schema.Number,
  updatedAt: Schema.Number,
  assignee: TaskAssignmentAssigneeSchema,
  workerState: WorkerStateSchema,
})
export type TaskAssignmentRow = typeof TaskAssignmentRowSchema.Type

export const TaskAssignmentStateSchema = Schema.Struct({
  orderedTaskIds: Schema.Array(Schema.String),
  rows: Schema.ReadonlyMap({ key: Schema.String, value: TaskAssignmentRowSchema }),
  workerActivityByForkId: Schema.ReadonlyMap({ key: Schema.String, value: WorkerActivitySchema }),
})
export type TaskAssignmentState = typeof TaskAssignmentStateSchema.Type

function flattenTaskTree(state: TaskGraphState): {
  orderedTaskIds: string[]
  depthByTaskId: ReadonlyMap<string, number>
} {
  const orderedTaskIds: string[] = []
  const depthByTaskId = new Map<string, number>()

  const visit = (taskId: string, depth: number) => {
    const task = state.tasks.get(taskId)
    if (!task) return
    orderedTaskIds.push(taskId)
    depthByTaskId.set(taskId, depth)
    for (const childId of task.childIds) {
      visit(childId, depth + 1)
    }
  }

  for (const rootTaskId of state.rootTaskIds) {
    visit(rootTaskId, 0)
  }

  return { orderedTaskIds, depthByTaskId }
}

function getToolTaskId(state: ToolState): Option.Option<string> {
  if (!('taskId' in state)) return Option.none()
  if (!Option.isOption(state.taskId)) return Option.none()
  return Option.flatMap(state.taskId, (taskId) =>
    typeof taskId === 'string' && taskId.length > 0 ? Option.some(taskId) : Option.none(),
  )
}

function isActiveWorkerTool(handle: ToolHandleFromSchema): boolean {
  return handle.state.phase === 'streaming' || handle.state.phase === 'executing'
}

function getRootToolHandles(toolState: { forks: ReadonlyMap<string | null, HarnessTurnState> }): Record<string, ToolHandleFromSchema> {
  const rootFork = toolState.forks.get(null)
  return rootFork ? getToolHandlesRecord(rootFork) : {}
}

function findActiveToolCallId(
  toolHandles: Record<string, ToolHandleFromSchema>,
  toolKey: 'spawnWorker' | 'killWorker',
  taskId: string,
): Option.Option<string> {
  for (const [toolCallId, handle] of Object.entries(toolHandles)) {
    if (handle.toolKey !== toolKey) continue
    if (!isActiveWorkerTool(handle)) continue
    const matchesTask = Option.match(getToolTaskId(handle.state), {
      onNone: () => false,
      onSome: (activeTaskId) => activeTaskId === taskId,
    })
    if (!matchesTask) continue
    return Option.some(toolCallId)
  }

  return Option.none()
}

function deriveWorkerState(args: {
  task: TaskRecord
  toolHandles: Record<string, ToolHandleFromSchema>
  agentState: AgentLifecycleState
  activityByForkId: ReadonlyMap<string, WorkerActivity>
}): WorkerState {
  const { task, toolHandles, agentState, activityByForkId } = args

  const activeSpawnToolCallId = findActiveToolCallId(toolHandles, 'spawnWorker', task.id)
  if (Option.isSome(activeSpawnToolCallId)) {
    const toolCallId = activeSpawnToolCallId.value
    const handle = toolHandles[toolCallId]
    const spawnRole = handle?.toolKey === 'spawnWorker'
      ? handle.state.role
      : Option.none<string>()
    return { status: 'spawning', toolCallId, role: spawnRole }
  }

  if (task.worker) {
    const activeKillToolCallId = findActiveToolCallId(toolHandles, 'killWorker', task.id)
    if (Option.isSome(activeKillToolCallId)) {
      return {
        status: 'killing',
        forkId: task.worker.forkId,
        toolCallId: activeKillToolCallId.value,
      }
    }

    const linkedAgent = getAgentByForkId(agentState, task.worker.forkId)
    const activity = activityByForkId.get(task.worker.forkId)

    if (linkedAgent?.status === 'working') {
      const activeSince = activity
        ? Option.getOrElse(activity.activeSince, () => task.updatedAt)
        : task.updatedAt
      return {
        status: 'working',
        forkId: task.worker.forkId,
        activeSince,
        accumulatedMs: activity?.accumulatedMs ?? 0,
        resumeCount: activity?.resumeCount ?? 0,
      }
    }

    if (linkedAgent || activity) {
      return {
        status: 'idle',
        forkId: task.worker.forkId,
        accumulatedMs: activity?.accumulatedMs ?? 0,
        completedAt: activity?.completedAt ?? Option.none(),
        resumeCount: activity?.resumeCount ?? 0,
      }
    }
  }

  return { status: 'unassigned' }
}

function deriveTaskAssignmentAssignee(task: TaskRecord): TaskAssignmentAssignee {
  if (task.worker) {
    return {
      kind: 'worker',
      role: task.worker.role,
      agentId: task.worker.agentId,
      forkId: task.worker.forkId,
    }
  }

  if (task.assignee === 'user') return { kind: 'user' }
  return { kind: 'none' }
}

function recomputeState(args: {
  taskGraph: TaskGraphState
  agentState: AgentLifecycleState
  toolState: { forks: ReadonlyMap<string | null, HarnessTurnState> }
  workerActivityByForkId: ReadonlyMap<string, WorkerActivity>
}): Pick<TaskAssignmentState, 'orderedTaskIds' | 'rows'> {
  const { orderedTaskIds, depthByTaskId } = flattenTaskTree(args.taskGraph)
  const toolHandles = getRootToolHandles(args.toolState)
  const rows = new Map<string, TaskAssignmentRow>()

  for (const taskId of orderedTaskIds) {
    const task = args.taskGraph.tasks.get(taskId)
    if (!task) continue

    rows.set(taskId, {
      taskId: task.id,
      title: task.title,
      status: task.status,
      parentId: task.parentId === null ? Option.none() : Option.some(task.parentId),
      depth: depthByTaskId.get(taskId) ?? 0,
      updatedAt: task.updatedAt,
      assignee: deriveTaskAssignmentAssignee(task),
      workerState: deriveWorkerState({
        task,
        toolHandles,
        agentState: args.agentState,
        activityByForkId: args.workerActivityByForkId,
      }),
    })
  }

  return { orderedTaskIds, rows }
}

function updateWorkerActivity(
  activityByForkId: ReadonlyMap<string, WorkerActivity>,
  forkId: string,
  updater: (current: Option.Option<WorkerActivity>) => Option.Option<WorkerActivity>,
): ReadonlyMap<string, WorkerActivity> {
  const currentActivity = activityByForkId.get(forkId)
  const current = currentActivity === undefined
    ? Option.none<WorkerActivity>()
    : Option.some(currentActivity)
  const next = updater(current)
  const result = new Map(activityByForkId)

  if (Option.isSome(next)) result.set(forkId, next.value)
  else result.delete(forkId)

  return result
}

function ensureWorkerActivity(
  activityByForkId: ReadonlyMap<string, WorkerActivity>,
  forkId: string,
): ReadonlyMap<string, WorkerActivity> {
  return updateWorkerActivity(activityByForkId, forkId, (current) =>
    Option.match(current, {
      onNone: () => Option.some({
        forkId,
        activeSince: Option.none(),
        accumulatedMs: 0,
        lastStintMs: 0,
        completedAt: Option.none(),
        resumeCount: 0,
      }),
      onSome: Option.some,
    }),
  )
}

function markWorkerWorking(
  activityByForkId: ReadonlyMap<string, WorkerActivity>,
  forkId: string,
  timestamp: number,
): ReadonlyMap<string, WorkerActivity> {
  return updateWorkerActivity(activityByForkId, forkId, (current) => {
    if (Option.isNone(current)) {
      return Option.some({
        forkId,
        activeSince: Option.some(timestamp),
        accumulatedMs: 0,
        lastStintMs: 0,
        completedAt: Option.none(),
        resumeCount: 0,
      })
    }

    const existing = current.value
    if (Option.isSome(existing.activeSince)) {
      return Option.some({
        ...existing,
        completedAt: Option.none(),
      })
    }

    // Only increment resumeCount when resuming from idle (completedAt was set)
    const isResume = Option.isSome(existing.completedAt)
    return Option.some({
      ...existing,
      activeSince: Option.some(timestamp),
      completedAt: Option.none(),
      resumeCount: isResume ? existing.resumeCount + 1 : existing.resumeCount,
      lastStintMs: existing.lastStintMs ?? 0,
    })
  })
}

function markWorkerIdle(
  activityByForkId: ReadonlyMap<string, WorkerActivity>,
  forkId: string,
  timestamp: number,
): ReadonlyMap<string, WorkerActivity> {
  return updateWorkerActivity(activityByForkId, forkId, (current) => {
    if (Option.isNone(current)) {
      return Option.some({
        forkId,
        activeSince: Option.none(),
        accumulatedMs: 0,
        lastStintMs: 0,
        completedAt: Option.some(timestamp),
        resumeCount: 0,
      })
    }

    const activeDelta = Option.match(current.value.activeSince, {
      onNone: () => 0,
      onSome: (activeSince) => Math.max(0, timestamp - activeSince),
    })

    return Option.some({
      forkId,
      activeSince: Option.none(),
      accumulatedMs: current.value.accumulatedMs + activeDelta,
      lastStintMs: activeDelta,
      completedAt: Option.some(timestamp),
      resumeCount: current.value.resumeCount,
    })
  })
}

function removeWorkerActivity(
  activityByForkId: ReadonlyMap<string, WorkerActivity>,
  forkId: string,
): ReadonlyMap<string, WorkerActivity> {
  return updateWorkerActivity(activityByForkId, forkId, () => Option.none())
}

type TaskAssignmentReads = readonly [
  typeof TaskGraphProjection,
  typeof AgentLifecycleProjection,
  typeof HarnessStateProjection
]

function rebuild({
  state,
  read,
  workerActivityByForkId = state.workerActivityByForkId,
}: {
  state: TaskAssignmentState
  read: ReadFn<TaskAssignmentReads>
  workerActivityByForkId?: ReadonlyMap<string, WorkerActivity>
}): TaskAssignmentState {
  const taskGraph = read(TaskGraphProjection)
  const agentState = read(AgentLifecycleProjection)
  const toolState = read(HarnessStateProjection)

  const next = recomputeState({
    taskGraph,
    agentState,
    toolState,
    workerActivityByForkId,
  })

  return {
    orderedTaskIds: next.orderedTaskIds,
    rows: next.rows,
    workerActivityByForkId,
  }
}

export const TaskAssignmentProjection = Projection.define<AppEvent>()({
  name: 'TaskAssignment',
  state: TaskAssignmentStateSchema,

  reads: [TaskGraphProjection, AgentLifecycleProjection, HarnessStateProjection] as const,

  initial: {
    orderedTaskIds: [],
    rows: new Map<string, TaskAssignmentRow>(),
    workerActivityByForkId: new Map<string, WorkerActivity>(),
  },

  eventHandlers: {
    agent_created: ({ event, state, read }) =>
      rebuild({ state, read, workerActivityByForkId: ensureWorkerActivity(state.workerActivityByForkId, event.forkId) }),

    turn_started: ({ event, state, read }) => {
      if (event.forkId === null) return rebuild({ state, read })
      return rebuild({ state, read, workerActivityByForkId: markWorkerWorking(state.workerActivityByForkId, event.forkId, event.timestamp) })
    },

    turn_outcome: ({ event, state, read }) => {
      if (event.forkId === null) return rebuild({ state, read })
      if (outcomeWillChainContinue(event.outcome)) {
        return rebuild({ state, read })
      }
      return rebuild({ state, read, workerActivityByForkId: markWorkerIdle(state.workerActivityByForkId, event.forkId, event.timestamp) })
    },

    interrupt: ({ event, state, read }) => {
      if (event.forkId === null) return rebuild({ state, read })
      return rebuild({ state, read, workerActivityByForkId: markWorkerIdle(state.workerActivityByForkId, event.forkId, event.timestamp) })
    },

    agent_killed: ({ event, state, read }) =>
      rebuild({ state, read, workerActivityByForkId: removeWorkerActivity(state.workerActivityByForkId, event.forkId) }),

    worker_user_killed: ({ event, state, read }) =>
      rebuild({ state, read, workerActivityByForkId: removeWorkerActivity(state.workerActivityByForkId, event.forkId) }),

    worker_idle_closed: ({ event, state, read }) =>
      rebuild({ state, read, workerActivityByForkId: removeWorkerActivity(state.workerActivityByForkId, event.forkId) }),

    task_created: ({ state, read }) => rebuild({ state, read }),
    task_updated: ({ state, read }) => rebuild({ state, read }),
    task_assigned: ({ state, read }) => rebuild({ state, read }),
    task_cancelled: ({ state, read }) => rebuild({ state, read }),
    tool_event: ({ state, read }) => rebuild({ state, read }),
    agent_task_changed: ({ state, read }) => rebuild({ state, read }),
  },
})
