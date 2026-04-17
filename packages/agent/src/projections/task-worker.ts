import { Projection, type ReadFn, type ForkedState } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import {
  AgentStatusProjection,
  getAgentByForkId,
  type AgentInfo,
  type AgentStatusState,
} from './agent-status'
import {
  TaskGraphProjection,
  type TaskGraphState,
  type TaskRecord,
  type TaskStatus,
} from './task-graph'
import { ToolStateProjection, type ToolStateProjectionState } from './tool-state'
import type { ToolHandle } from '../tools/tool-handle'
import type { ToolState } from '../models/tool-state'

export type WorkerState =
  | { readonly status: 'unassigned' }
  | { readonly status: 'spawning'; readonly toolCallId: string; readonly role: string | null }
  | {
      readonly status: 'working'
      readonly forkId: string
      readonly activeSince: number
      readonly accumulatedMs: number
      readonly resumeCount: number
    }
  | {
      readonly status: 'idle'
      readonly forkId: string
      readonly accumulatedMs: number
      readonly completedAt: number | null
      readonly resumeCount: number
    }
  | {
      readonly status: 'killing'
      readonly forkId: string
      readonly toolCallId: string
    }

export interface WorkerActivity {
  readonly forkId: string
  readonly activeSince: number | null
  readonly accumulatedMs: number
  readonly completedAt: number | null
  readonly resumeCount: number
}

export type TaskWorkerAssignee =
  | { readonly kind: 'none' }
  | { readonly kind: 'user' }
  | {
      readonly kind: 'worker'
      readonly role: string
      readonly agentId: string
      readonly forkId: string
    }

export interface TaskWorkerSnapshot {
  readonly taskId: string
  readonly title: string
  readonly status: TaskStatus
  readonly parentId: string | null
  readonly depth: number
  readonly updatedAt: number
  readonly assignee: TaskWorkerAssignee
  readonly workerState: WorkerState
}

export interface TaskWorkerState {
  readonly orderedTaskIds: readonly string[]
  readonly snapshots: ReadonlyMap<string, TaskWorkerSnapshot>
  readonly workerActivityByForkId: ReadonlyMap<string, WorkerActivity>
}

const ACTIVE_TOOL_PHASES = new Set(['streaming', 'executing'] as const)

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

function getToolTaskId(state: ToolState): string | null {
  if (!('id' in state)) return null
  return typeof state.id === 'string' && state.id.length > 0 ? state.id : null
}

function isActiveWorkerTool(handle: ToolHandle): boolean {
  return ACTIVE_TOOL_PHASES.has(handle.state.phase as 'streaming' | 'executing')
}

function getRootToolHandles(toolState: { forks: ReadonlyMap<string | null, ToolStateProjectionState> }): Record<string, ToolHandle> {
  return toolState.forks.get(null)?.toolHandles ?? {}
}

function findActiveToolCallId(
  toolHandles: Record<string, ToolHandle>,
  toolKey: 'spawnWorker' | 'killWorker',
  taskId: string,
): string | null {
  for (const [toolCallId, handle] of Object.entries(toolHandles)) {
    if (handle.toolKey !== toolKey) continue
    if (!isActiveWorkerTool(handle)) continue
    if (getToolTaskId(handle.state) !== taskId) continue
    return toolCallId
  }

  return null
}

function deriveWorkerState(args: {
  task: TaskRecord
  toolHandles: Record<string, ToolHandle>
  agentState: AgentStatusState
  activityByForkId: ReadonlyMap<string, WorkerActivity>
}): WorkerState {
  const { task, toolHandles, agentState, activityByForkId } = args

  const activeSpawnToolCallId = findActiveToolCallId(toolHandles, 'spawnWorker', task.id)
  if (activeSpawnToolCallId) {
    const handle = toolHandles[activeSpawnToolCallId]
    return { status: 'spawning', toolCallId: activeSpawnToolCallId, role: 'worker' as const }
  }

  if (task.worker) {
    const activeKillToolCallId = findActiveToolCallId(toolHandles, 'killWorker', task.id)
    if (activeKillToolCallId) {
      return {
        status: 'killing',
        forkId: task.worker.forkId,
        toolCallId: activeKillToolCallId,
      }
    }

    const linkedAgent = getAgentByForkId(agentState, task.worker.forkId)
    const activity = activityByForkId.get(task.worker.forkId)

    if (linkedAgent?.status === 'working') {
      return {
        status: 'working',
        forkId: task.worker.forkId,
        activeSince: activity?.activeSince ?? task.updatedAt,
        accumulatedMs: activity?.accumulatedMs ?? 0,
        resumeCount: activity?.resumeCount ?? 0,
      }
    }

    if (linkedAgent || activity) {
      return {
        status: 'idle',
        forkId: task.worker.forkId,
        accumulatedMs: activity?.accumulatedMs ?? 0,
        completedAt: activity?.completedAt ?? null,
        resumeCount: activity?.resumeCount ?? 0,
      }
    }
  }

  return { status: 'unassigned' }
}

function deriveTaskWorkerAssignee(task: TaskRecord): TaskWorkerAssignee {
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
  agentState: AgentStatusState
  toolState: { forks: ReadonlyMap<string | null, ToolStateProjectionState> }
  workerActivityByForkId: ReadonlyMap<string, WorkerActivity>
}): Pick<TaskWorkerState, 'orderedTaskIds' | 'snapshots'> {
  const { orderedTaskIds, depthByTaskId } = flattenTaskTree(args.taskGraph)
  const toolHandles = getRootToolHandles(args.toolState)
  const snapshots = new Map<string, TaskWorkerSnapshot>()

  for (const taskId of orderedTaskIds) {
    const task = args.taskGraph.tasks.get(taskId)
    if (!task) continue

    snapshots.set(taskId, {
      taskId: task.id,
      title: task.title,
      status: task.status,
      parentId: task.parentId,
      depth: depthByTaskId.get(taskId) ?? 0,
      updatedAt: task.updatedAt,
      assignee: deriveTaskWorkerAssignee(task),
      workerState: deriveWorkerState({
        task,
        toolHandles,
        agentState: args.agentState,
        activityByForkId: args.workerActivityByForkId,
      }),
    })
  }

  return { orderedTaskIds, snapshots }
}

function updateWorkerActivity(
  activityByForkId: ReadonlyMap<string, WorkerActivity>,
  forkId: string,
  updater: (current: WorkerActivity | null) => WorkerActivity | null,
): ReadonlyMap<string, WorkerActivity> {
  const current = activityByForkId.get(forkId) ?? null
  const next = updater(current)
  const result = new Map(activityByForkId)

  if (next) result.set(forkId, next)
  else result.delete(forkId)

  return result
}

function ensureWorkerActivity(
  activityByForkId: ReadonlyMap<string, WorkerActivity>,
  forkId: string,
): ReadonlyMap<string, WorkerActivity> {
  return updateWorkerActivity(activityByForkId, forkId, (current) => current ?? {
    forkId,
    activeSince: null,
    accumulatedMs: 0,
    completedAt: null,
    resumeCount: 0,
  })
}

function markWorkerWorking(
  activityByForkId: ReadonlyMap<string, WorkerActivity>,
  forkId: string,
  timestamp: number,
): ReadonlyMap<string, WorkerActivity> {
  return updateWorkerActivity(activityByForkId, forkId, (current) => {
    if (current && current.activeSince !== null) {
      return {
        ...current,
        completedAt: null,
      }
    }

    // Only increment resumeCount when resuming from idle (completedAt was set)
    const isResume = current?.completedAt !== null && current?.completedAt !== undefined
    return {
      forkId,
      activeSince: timestamp,
      accumulatedMs: current?.accumulatedMs ?? 0,
      completedAt: null,
      resumeCount: isResume ? (current?.resumeCount ?? 0) + 1 : (current?.resumeCount ?? 0),
    }
  })
}

function markWorkerIdle(
  activityByForkId: ReadonlyMap<string, WorkerActivity>,
  forkId: string,
  timestamp: number,
): ReadonlyMap<string, WorkerActivity> {
  return updateWorkerActivity(activityByForkId, forkId, (current) => {
    if (!current) {
      return {
        forkId,
        activeSince: null,
        accumulatedMs: 0,
        completedAt: timestamp,
        resumeCount: 0,
      }
    }

    const activeDelta = current.activeSince === null ? 0 : Math.max(0, timestamp - current.activeSince)

    return {
      forkId,
      activeSince: null,
      accumulatedMs: current.accumulatedMs + activeDelta,
      completedAt: timestamp,
      resumeCount: current.resumeCount,
    }
  })
}

function removeWorkerActivity(
  activityByForkId: ReadonlyMap<string, WorkerActivity>,
  forkId: string,
): ReadonlyMap<string, WorkerActivity> {
  return updateWorkerActivity(activityByForkId, forkId, () => null)
}

type TaskWorkerReads = readonly [
  typeof TaskGraphProjection,
  typeof AgentStatusProjection,
  typeof ToolStateProjection
]

function rebuild({
  state,
  read,
  workerActivityByForkId = state.workerActivityByForkId,
}: {
  state: TaskWorkerState
  read: ReadFn<TaskWorkerReads>
  workerActivityByForkId?: ReadonlyMap<string, WorkerActivity>
}): TaskWorkerState {
  const taskGraph = read(TaskGraphProjection)
  const agentState = read(AgentStatusProjection)
  const toolState = read(ToolStateProjection)

  const next = recomputeState({
    taskGraph,
    agentState,
    toolState,
    workerActivityByForkId,
  })

  return {
    orderedTaskIds: next.orderedTaskIds,
    snapshots: next.snapshots,
    workerActivityByForkId,
  }
}

export const TaskWorkerProjection = Projection.define<AppEvent, TaskWorkerState>()({
  name: 'TaskWorker',

  reads: [TaskGraphProjection, AgentStatusProjection, ToolStateProjection] as const,

  initial: {
    orderedTaskIds: [],
    snapshots: new Map(),
    workerActivityByForkId: new Map(),
  },

  eventHandlers: {
    agent_created: ({ event, state, read }) =>
      rebuild({ state, read, workerActivityByForkId: ensureWorkerActivity(state.workerActivityByForkId, event.forkId) }),

    turn_started: ({ event, state, read }) => {
      if (event.forkId === null) return rebuild({ state, read })
      return rebuild({ state, read, workerActivityByForkId: markWorkerWorking(state.workerActivityByForkId, event.forkId, event.timestamp) })
    },

    turn_completed: ({ event, state, read }) => {
      if (event.forkId === null) return rebuild({ state, read })
      if (event.result.success && event.result.turnDecision === 'continue') {
        return rebuild({ state, read })
      }
      return rebuild({ state, read, workerActivityByForkId: markWorkerIdle(state.workerActivityByForkId, event.forkId, event.timestamp) })
    },

    turn_unexpected_error: ({ event, state, read }) => {
      if (event.forkId === null) return rebuild({ state, read })
      return rebuild({ state, read, workerActivityByForkId: markWorkerIdle(state.workerActivityByForkId, event.forkId, event.timestamp) })
    },

    interrupt: ({ event, state, read }) => {
      if (event.forkId === null) return rebuild({ state, read })
      return rebuild({ state, read, workerActivityByForkId: markWorkerIdle(state.workerActivityByForkId, event.forkId, event.timestamp) })
    },

    agent_killed: ({ event, state, read }) =>
      rebuild({ state, read, workerActivityByForkId: removeWorkerActivity(state.workerActivityByForkId, event.forkId) }),

    subagent_user_killed: ({ event, state, read }) =>
      rebuild({ state, read, workerActivityByForkId: removeWorkerActivity(state.workerActivityByForkId, event.forkId) }),

    subagent_idle_closed: ({ event, state, read }) =>
      rebuild({ state, read, workerActivityByForkId: removeWorkerActivity(state.workerActivityByForkId, event.forkId) }),

    task_created: ({ state, read }) => rebuild({ state, read }),
    task_updated: ({ state, read }) => rebuild({ state, read }),
    task_assigned: ({ state, read }) => rebuild({ state, read }),
    task_cancelled: ({ state, read }) => rebuild({ state, read }),
    tool_event: ({ state, read }) => rebuild({ state, read }),
  },
})
