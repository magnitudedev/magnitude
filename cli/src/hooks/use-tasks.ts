import { useEffect, useRef, useState } from 'react'
import type { TaskWorkerSnapshot, TaskWorkerState } from '@magnitudedev/agent'
import type {
  GhostOverlay,
  TaskDisplayRow,
  TaskListItem,
  VisibleWorkerContinuity,
} from '../components/chat/task-list/index'
import { deriveTaskDisplayRow } from '../components/chat/task-list/index'

type AgentClientLike = {
  state: {
    taskWorker: {
      subscribe: (cb: (state: TaskWorkerState) => void) => () => void
    }
  }
}

type UseTasksArgs = {
  client: AgentClientLike | null
}

type TaskListDisplayState = {
  rowsByTaskId: ReadonlyMap<string, TaskDisplayRow>
  orderedTaskIds: readonly string[]
  ghostsByTaskId: ReadonlyMap<string, GhostOverlay>
  previousContinuityByTaskId: ReadonlyMap<string, VisibleWorkerContinuity>
  nextGhostExpiryAt: number | null
  items: TaskListItem[]
}

const EMPTY_TASK_WORKER_STATE: TaskWorkerState = {
  orderedTaskIds: [],
  snapshots: new Map(),
  workerActivityByForkId: new Map(),
}

const EMPTY_DISPLAY_STATE: TaskListDisplayState = {
  rowsByTaskId: new Map(),
  orderedTaskIds: [],
  ghostsByTaskId: new Map(),
  previousContinuityByTaskId: new Map(),
  nextGhostExpiryAt: null,
  items: [],
}

const KILLED_GHOST_HOLD_MS = 1000

function pruneExpiredGhosts(
  ghostsByTaskId: ReadonlyMap<string, GhostOverlay>,
  now: number,
): {
  ghostsByTaskId: ReadonlyMap<string, GhostOverlay>
  nextGhostExpiryAt: number | null
} {
  let nextGhostExpiryAt: number | null = null
  let changed = false
  const nextGhostsByTaskId = new Map<string, GhostOverlay>()

  for (const [taskId, ghost] of ghostsByTaskId) {
    if (ghost.expiresAt <= now) {
      changed = true
      continue
    }

    nextGhostsByTaskId.set(taskId, ghost)
    nextGhostExpiryAt = nextGhostExpiryAt === null
      ? ghost.expiresAt
      : Math.min(nextGhostExpiryAt, ghost.expiresAt)
  }

  return {
    ghostsByTaskId: changed ? nextGhostsByTaskId : ghostsByTaskId,
    nextGhostExpiryAt,
  }
}

function getVisibleWorkerContinuity(
  row: TaskDisplayRow,
  index: number,
): VisibleWorkerContinuity {
  const assignee = row.assignee
  if (assignee.kind !== 'worker' || !assignee.ghostEligible || !assignee.continuityKey) {
    return null
  }

  return {
    continuityKey: assignee.continuityKey,
    taskId: row.taskId,
    label: assignee.label,
    lastKnownIndex: index,
  }
}

function mergeDisplayRows(
  orderedTaskIds: readonly string[],
  rowsByTaskId: ReadonlyMap<string, TaskDisplayRow>,
  ghostsByTaskId: ReadonlyMap<string, GhostOverlay>,
): TaskListItem[] {
  return orderedTaskIds
    .map((taskId) => {
      const row = rowsByTaskId.get(taskId)
      if (!row) return null

      const ghost = ghostsByTaskId.get(taskId) ?? null
      return ghost
        ? {
            ...row,
            assignee: {
              kind: 'ghost',
              icon: '✕',
              label: ghost.label,
              tone: 'ghost',
              expiresAt: ghost.expiresAt,
            },
          }
        : row
    })
    .filter((row): row is TaskDisplayRow => Boolean(row))
}

function reconcileDisplayState(
  taskWorkerState: TaskWorkerState,
  previousDisplayState: TaskListDisplayState,
  now: number,
): TaskListDisplayState {
  const derivedRows = taskWorkerState.orderedTaskIds
    .map((taskId) => taskWorkerState.snapshots.get(taskId))
    .filter((snapshot): snapshot is TaskWorkerSnapshot => Boolean(snapshot))
    .map((snapshot) => deriveTaskDisplayRow(snapshot))

  const rowsByTaskId = new Map<string, TaskDisplayRow>(
    derivedRows.map(row => [row.taskId, row]),
  )
  const orderedTaskIds = derivedRows.map(row => row.taskId)

  const prunedGhosts = pruneExpiredGhosts(previousDisplayState.ghostsByTaskId, now)
  const nextGhostsByTaskId = new Map(prunedGhosts.ghostsByTaskId)
  const nextContinuityByTaskId = new Map<string, VisibleWorkerContinuity>()

  for (const [index, row] of derivedRows.entries()) {
    const continuity = getVisibleWorkerContinuity(row, index)
    nextContinuityByTaskId.set(row.taskId, continuity)

    const assignee = row.assignee
    const hasLiveWorker =
      assignee.kind === 'worker'
      && assignee.ghostEligible
      && assignee.continuityKey !== null

    if (hasLiveWorker) {
      nextGhostsByTaskId.delete(row.taskId)
    }
  }

  for (const [taskId, previousContinuity] of previousDisplayState.previousContinuityByTaskId) {
    if (!previousContinuity) continue

    const nextContinuity = nextContinuityByTaskId.get(taskId) ?? null
    const workerRemoved =
      nextContinuity === null
      || nextContinuity.continuityKey !== previousContinuity.continuityKey

    if (!workerRemoved) continue

    if (!rowsByTaskId.has(taskId)) continue

    nextGhostsByTaskId.set(taskId, {
      taskId,
      expiresAt: now + KILLED_GHOST_HOLD_MS,
      label: previousContinuity.label,
    })
  }

  const { nextGhostExpiryAt } = pruneExpiredGhosts(nextGhostsByTaskId, now)

  return {
    rowsByTaskId,
    orderedTaskIds,
    ghostsByTaskId: nextGhostsByTaskId,
    previousContinuityByTaskId: nextContinuityByTaskId,
    nextGhostExpiryAt,
    items: mergeDisplayRows(orderedTaskIds, rowsByTaskId, nextGhostsByTaskId),
  }
}

export function useTasks({ client }: UseTasksArgs): TaskListItem[] {
  const [displayState, setDisplayState] = useState<TaskListDisplayState>(EMPTY_DISPLAY_STATE)
  const latestTaskWorkerStateRef = useRef<TaskWorkerState>(EMPTY_TASK_WORKER_STATE)
  const latestDisplayStateRef = useRef<TaskListDisplayState>(EMPTY_DISPLAY_STATE)

  useEffect(() => {
    if (!client) {
      latestTaskWorkerStateRef.current = EMPTY_TASK_WORKER_STATE
      latestDisplayStateRef.current = EMPTY_DISPLAY_STATE
      setDisplayState(EMPTY_DISPLAY_STATE)
      return
    }

    return client.state.taskWorker.subscribe((state) => {
      latestTaskWorkerStateRef.current = state
      setDisplayState((previousDisplayState) => {
        const nextDisplayState = reconcileDisplayState(state, previousDisplayState, Date.now())
        latestDisplayStateRef.current = nextDisplayState
        return nextDisplayState
      })
    })
  }, [client])

  useEffect(() => {
    if (displayState.nextGhostExpiryAt === null) return

    const timeoutMs = Math.max(0, displayState.nextGhostExpiryAt - Date.now())
    const timeout = globalThis.setTimeout(() => {
      setDisplayState((previousDisplayState) => {
        const nextDisplayState = reconcileDisplayState(
          latestTaskWorkerStateRef.current,
          previousDisplayState,
          Date.now(),
        )
        latestDisplayStateRef.current = nextDisplayState
        return nextDisplayState
      })
    }, timeoutMs)

    return () => globalThis.clearTimeout(timeout)
  }, [displayState.nextGhostExpiryAt])

  return displayState.items
}