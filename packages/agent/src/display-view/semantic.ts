import { Option } from 'effect'
import {
  forkIdToKey,
  type DisplayAgent,
  type DisplayActor,
  type DisplayRootStatus,
  type DisplayWorkerStatus,
  type DisplayState,
  type TaskAssignee,
  type TaskDisplayRow,
} from '@magnitudedev/acn-protocol'
import { DEFAULT_CHAT_NAME } from '../constants'
import type { AgentLifecycleState, AgentInfo } from '../projections/agent-lifecycle'
import type { ActiveModelRequests } from '../display/model-request-activity'
import type { TaskAssignmentRow, TaskAssignmentState, WorkerActivity } from '../projections/task-assignment'

function titleCase(value: string): string {
  return value.length === 0 ? value : value.charAt(0).toUpperCase() + value.slice(1)
}

const ROOT_ACTOR_KEY = forkIdToKey(null)

const idleActorStatus = (): DisplayWorkerStatus => ({
  phase: 'idle',
  activeSince: null,
  lastWorkMs: 0,
  accumulatedMs: 0,
  resumeCount: 0,
})

const materializeRootDetail = (
  rootWork: AgentLifecycleState['rootWork'],
  modelRequests: ActiveModelRequests,
): Extract<DisplayRootStatus, { readonly _tag: 'Working' }>['detail'] => {
  if (rootWork._isThinking) return { _tag: 'Thinking' }
  const request = modelRequests.get(ROOT_ACTOR_KEY)
  if (
    request?.phase === 'prefill'
    && request.turnId === rootWork._currentTurn?.turnId
  ) {
    return {
      _tag: 'Prefill',
      completedTokens: Math.max(0, Math.floor(request.completedTokens ?? 0)),
      totalTokens: Math.max(0, Math.floor(request.totalTokens ?? 0)),
      cachedTokens: Math.max(0, Math.floor(request.cachedTokens ?? 0)),
    }
  }
  if (rootWork._currentTurn !== null && !rootWork._currentTurn.modelActivityStarted) {
    return {
      _tag: 'WaitingForModel',
      turnStartedAt: rootWork._currentTurn.startedAt,
    }
  }
  return { _tag: 'NoDetail' }
}

const materializeRootStatus = (
  rootWork: AgentLifecycleState['rootWork'],
  modelRequests: ActiveModelRequests,
): DisplayRootStatus => {
  switch (rootWork.phase) {
    case 'idle':
      return { _tag: 'Idle' }
    case 'active':
      if (rootWork.chainStartedAt === null) {
        throw new Error('active root work requires chainStartedAt')
      }
      return {
        _tag: 'Working',
        chainStartedAt: rootWork.chainStartedAt,
        detail: materializeRootDetail(rootWork, modelRequests),
        activeChildCount: rootWork.activeChildCount,
      }
    case 'worked':
      return { _tag: 'Worked', lastProductiveMs: rootWork.lastProductiveMs }
    case 'interrupted':
      return { _tag: 'Interrupted', lastProductiveMs: rootWork.lastProductiveMs }
  }
}

/**
 * Worker actor status derived from AgentInfo (phase from status + lastIdleReason)
 * and WorkerActivity (timer from TaskAssignmentProjection).
 */
const deriveWorkerStatus = (
  agent: AgentInfo,
  activity: WorkerActivity | undefined,
): DisplayWorkerStatus => {
  const phase: DisplayWorkerStatus['phase'] =
    agent.status === 'working' ? 'working'
    : agent.lastIdleReason === 'interrupt' ? 'interrupted'
    : agent.status === 'idle' ? 'worked'
    : 'idle'

  const activeSince = activity && Option.isSome(activity.activeSince)
    ? activity.activeSince.value
    : null

  return {
    phase,
    activeSince,
    lastWorkMs: activity?.lastStintMs ?? 0,
    accumulatedMs: activity?.accumulatedMs ?? 0,
    resumeCount: activity?.resumeCount ?? 0,
  }
}

const materializeActorContext = (
  forkId: string | null,
  windowState: { readonly forks: ReadonlyMap<string | null, { readonly tokenEstimate: number }> },
  compactionState: { readonly forks: ReadonlyMap<string | null, { readonly _tag: string }> },
): DisplayActor['context'] => ({
  tokenEstimate: windowState.forks.get(forkId)?.tokenEstimate ?? 0,
  isCompacting: compactionState.forks.get(forkId)?._tag === 'compacting',
})

export const materializeDisplayAgents = (agentStatus: AgentLifecycleState): Record<string, DisplayAgent> => {
  const agents: Record<string, DisplayAgent> = {}
  for (const agent of agentStatus.agents.values()) {
    agents[forkIdToKey(agent.forkId)] = {
      name: agent.name,
      role: agent.role,
      status: Option.some(agent.status),
    }
  }
  return agents
}

export const materializeDisplayActors = (
  agentStatus: AgentLifecycleState,
  taskWorker: TaskAssignmentState,
  windowState: { readonly forks: ReadonlyMap<string | null, { readonly tokenEstimate: number }> },
  compactionState: { readonly forks: ReadonlyMap<string | null, { readonly _tag: string }> },
  modelRequests: ActiveModelRequests,
): Record<string, DisplayActor> => {
  const actors: Record<string, DisplayActor> = {
    [ROOT_ACTOR_KEY]: {
      kind: 'root',
      name: 'Leader',
      role: 'leader',
      parentActorKey: null,
      taskId: null,
      status: materializeRootStatus(agentStatus.rootWork, modelRequests),
      context: materializeActorContext(null, windowState, compactionState),
    },
  }

  for (const agent of agentStatus.agents.values()) {
    const key = forkIdToKey(agent.forkId)
    const activity = taskWorker.workerActivityByForkId.get(agent.forkId)
    actors[key] = {
      kind: 'worker',
      name: agent.name,
      role: agent.role,
      parentActorKey: forkIdToKey(agent.parentForkId),
      taskId: agent.taskId,
      status: deriveWorkerStatus(agent, activity),
      context: materializeActorContext(agent.forkId, windowState, compactionState),
    }
  }

  for (const taskId of taskWorker.orderedTaskIds) {
    const row = taskWorker.rows.get(taskId)
    if (!row || row.assignee.kind !== 'worker') continue

    const key = forkIdToKey(row.assignee.forkId)
    if (actors[key]) continue

    actors[key] = {
      kind: 'worker',
      name: row.title,
      role: row.assignee.role,
      parentActorKey: ROOT_ACTOR_KEY,
      taskId: row.taskId,
      status: idleActorStatus(),
      context: materializeActorContext(row.assignee.forkId, windowState, compactionState),
    }
  }

  return actors
}

const materializeAssignee = (row: TaskAssignmentRow): TaskAssignee => {
  if (row.assignee.kind === 'user') {
    return { kind: 'user', label: 'user', tone: 'warning' }
  }

  if (row.workerState.status === 'spawning') {
    const role = row.workerState.role
    if (Option.isNone(role)) return { kind: 'none' }
    return {
      kind: 'worker',
      variant: 'spawning',
      label: titleCase(role.value),
      icon: '+',
      tone: 'active',
      interactiveForkId: Option.none(),
      timer: Option.none(),
      resumed: false,
      continuityKey: Option.none(),
      ghostEligible: false,
    }
  }

  if (row.assignee.kind !== 'worker') return { kind: 'none' }

  switch (row.workerState.status) {
    case 'working':
      return {
        kind: 'actor',
        actorKey: forkIdToKey(row.workerState.forkId),
        taskState: 'assigned',
        timer: Option.none(),
      }
    case 'idle':
      return {
        kind: 'actor',
        actorKey: forkIdToKey(row.workerState.forkId),
        taskState: 'assigned',
        timer: Option.none(),
      }
    case 'killing':
      return {
        kind: 'actor',
        actorKey: forkIdToKey(row.workerState.forkId),
        taskState: 'killing',
        timer: Option.none(),
      }
    case 'unassigned':
      return {
        kind: 'actor',
        actorKey: forkIdToKey(row.assignee.forkId),
        taskState: 'assigned',
        timer: Option.none(),
      }
  }
}

export const materializeDisplayTasks = (taskWorker: TaskAssignmentState): DisplayState['tasks'] => {
  const byId: Record<string, TaskDisplayRow> = {}
  const order: string[] = []
  let completedCount = 0

  for (const taskId of taskWorker.orderedTaskIds) {
    const row = taskWorker.rows.get(taskId)
    if (!row) continue

    if (row.status === 'completed') completedCount++

    byId[taskId] = {
      rowId: `task:${row.taskId}`,
      kind: 'task',
      taskId: row.taskId,
      title: row.title,
      status: row.status,
      parentId: row.parentId,
      depth: row.depth,
      updatedAt: row.updatedAt,
      assignee: materializeAssignee(row),
    }
    order.push(taskId)
  }

  const totalCount = order.length
  return {
    byId,
    order,
    summary: {
      totalCount,
      completedCount,
      incompleteCount: totalCount - completedCount,
    },
  }
}

export const materializeDisplaySession = (args: {
  readonly sessionId: string
  readonly title: string | null
  readonly cwd: string
}): DisplayState['session'] => ({
  sessionId: args.sessionId,
  title: args.title ?? DEFAULT_CHAT_NAME,
  cwd: args.cwd,
})
