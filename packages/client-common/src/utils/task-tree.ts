import type { TaskDisplayRow } from '@magnitudedev/sdk'
import { Option } from 'effect'

type TaskRecord = {
  id: string
  title: string
  parentId: string | null
  childIds: readonly string[]
  assignee: unknown
  worker: { agentId: string; forkId: string; role: string } | null
  status: 'pending' | 'completed'
  createdAt: number
  updatedAt: number
  completedAt: number | null
}

export type TaskGraphState = {
  tasks: ReadonlyMap<string, TaskRecord>
  rootTaskIds: readonly string[]
}

function countDescendants(state: TaskGraphState, taskId: string): number {
  const task = state.tasks.get(taskId)
  if (!task) return 0

  let count = 0
  const stack = [...task.childIds]
  while (stack.length > 0) {
    const currentId = stack.pop()
    if (!currentId) continue
    const current = state.tasks.get(currentId)
    if (!current) continue
    count += 1
    stack.push(...current.childIds)
  }

  return count
}

function toTaskDisplayRow(task: TaskRecord, depth: number): TaskDisplayRow {
  void countDescendants

  return {
    rowId: `task:${task.id}`,
    kind: 'task',
    taskId: task.id,
    title: task.title,

    status: task.status,
    depth,
    parentId: task.parentId === null ? Option.none() : Option.some(task.parentId),
    updatedAt: task.updatedAt,
    assignee: task.worker
      ? {
          kind: 'actor' as const,
          actorKey: task.worker.forkId,
          taskState: 'assigned' as const,
          timer: Option.none(),
        }
      : task.assignee === 'user'
        ? {
            kind: 'user' as const,
            label: 'user' as const,
            tone: 'warning' as const,
          }
        : { kind: 'none' as const },
  }
}

export function flattenTaskTree(state: TaskGraphState): TaskDisplayRow[] {
  const rows: TaskDisplayRow[] = []

  const visitChildren = (childIds: readonly string[], depth: number) => {
    for (const childId of childIds) visit(childId, depth)
  }

  const visit = (taskId: string, depth: number) => {
    const task = state.tasks.get(taskId)
    if (!task) return
    rows.push(toTaskDisplayRow(task, depth))
    visitChildren(task.childIds, depth + 1)
  }

  for (const rootId of state.rootTaskIds) visit(rootId, 0)

  return rows
}

export type RootSummary = {
  task: TaskDisplayRow
  startIndex: number
  endIndex: number
  completed: number
  active: number
  total: number
}

export function buildRootSummaries(tasks: readonly TaskDisplayRow[]): RootSummary[] {
  const rootStartIndexes: number[] = []
  for (let i = 0; i < tasks.length; i++) {
    const task = tasks[i]
    if (!task) continue
    if (task.kind !== 'task' || task.depth !== 0) continue
    rootStartIndexes.push(i)
  }

  const summaries: RootSummary[] = []
  for (let i = 0; i < rootStartIndexes.length; i++) {
    const startIndex = rootStartIndexes[i]
    if (startIndex === undefined) continue

    const nextStartIndex = rootStartIndexes[i + 1]
    const endIndex = nextStartIndex ?? tasks.length

    const rootTask = tasks[startIndex]
    if (!rootTask) continue

    let completed = 0
    let active = 0
    let total = 0
    for (let rowIndex = startIndex; rowIndex < endIndex; rowIndex++) {
      const rowTask = tasks[rowIndex]
      if (!rowTask || rowTask.kind !== 'task') continue

      total += 1
      if (rowTask.status === 'completed') {
        completed += 1
      } else {
        active += 1
      }
    }

    summaries.push({
      task: rootTask,
      startIndex,
      endIndex,
      completed,
      active,
      total,
    })
  }

  return summaries
}

export function findOwningRootIndex(tasks: readonly TaskDisplayRow[], rowIndex: number): number | null {
  if (rowIndex < 0 || rowIndex >= tasks.length) return null

  for (let i = rowIndex; i >= 0; i--) {
    const task = tasks[i]
    if (!task || task.kind !== 'task') continue
    if (task.depth !== 0) continue
    return i
  }

  return null
}
