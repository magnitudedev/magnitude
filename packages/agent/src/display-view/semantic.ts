import { Option } from 'effect'
import {
  forkIdToKey,
  type DisplayAgent,
  type DisplayActor,
  type DisplayActorWork,
  type DisplayActivity,
  type DisplayState,
  type TaskAssignee,
  type TaskDisplayRow,
} from '@magnitudedev/protocol'
import { DEFAULT_CHAT_NAME } from '../constants'
import type { AgentLifecycleState, AgentInfo } from '../projections/agent-lifecycle'
import type { TaskAssignmentRow, TaskAssignmentState, WorkerActivity } from '../projections/task-assignment'

function titleCase(value: string): string {
  return value.length === 0 ? value : value.charAt(0).toUpperCase() + value.slice(1)
}

const ROOT_ACTOR_KEY = forkIdToKey(null)

const idleActorWork = (): DisplayActorWork => ({
  phase: 'idle',
  activeSince: null,
  lastWorkMs: 0,
  accumulatedMs: 0,
  resumeCount: 0,
  activity: null,
  activeChildCount: 0,
})

/**
 * Root actor work from AgentLifecycleState.rootWork directly.
 * Root timer = current chain (chainStartedAt). Completed summary = lastChainMs.
 */
const materializeRootWork = (
  rootWork: AgentLifecycleState['rootWork'],
): DisplayActorWork => ({
  phase: rootWork.phase,
  activeSince: rootWork.chainStartedAt,
  lastWorkMs: rootWork.lastChainMs,
  accumulatedMs: rootWork.lastChainMs,
  resumeCount: 0,
  activity: rootWork.activity,
  activeChildCount: rootWork.activeChildCount,
})

/**
 * Worker actor work derived from AgentInfo (phase from status + lastIdleReason)
 * and WorkerActivity (timer from TaskAssignmentProjection).
 */
const deriveWorkerWork = (
  agent: AgentInfo,
  activity: WorkerActivity | undefined,
): DisplayActorWork => {
  const phase: DisplayActorWork['phase'] =
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
    activity: null,
    activeChildCount: 0,
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
): Record<string, DisplayActor> => {
  const actors: Record<string, DisplayActor> = {
    [ROOT_ACTOR_KEY]: {
      kind: 'root',
      name: 'Leader',
      role: 'leader',
      parentActorKey: null,
      taskId: null,
      work: materializeRootWork(agentStatus.rootWork),
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
      work: deriveWorkerWork(agent, activity),
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
      work: idleActorWork(),
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
