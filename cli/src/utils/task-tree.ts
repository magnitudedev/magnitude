import type { TaskListItem } from '../components/chat/types'

type TaskRecord = {
  id: string
  title: string
  taskType: string
  parentId: string | null
  childIds: readonly string[]
  assignee: unknown
  worker: { agentId: string; forkId: string; role: string } | null
  status: 'pending' | 'working' | 'completed'
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

function toTaskListItem(task: TaskRecord, depth: number): TaskListItem {
  return {
    taskId: task.id,
    title: task.title,
    type: task.taskType,
    status: task.status,
    depth,
    parentId: task.parentId,
    createdAt: task.createdAt,
    updatedAt: task.updatedAt,
    completedAt: task.completedAt,
    assignee: task.worker
      ? { kind: 'worker', workerType: task.worker.role, agentId: task.worker.agentId }
      : task.assignee === 'user'
        ? { kind: 'user' }
        : { kind: 'none' },
    workerForkId: task.worker?.forkId ?? null,
  }
}

export function flattenTaskTree(state: TaskGraphState): TaskListItem[] {
  const rows: TaskListItem[] = []

  const visitChildren = (childIds: readonly string[], depth: number) => {
    for (const childId of childIds) visit(childId, depth)
  }

  const visit = (taskId: string, depth: number) => {
    const task = state.tasks.get(taskId)
    if (!task) return
    rows.push(toTaskListItem(task, depth))
    visitChildren(task.childIds, depth + 1)
  }

  for (const rootId of state.rootTaskIds) visit(rootId, 0)

  return rows
}

export type RootSummary = {
  task: TaskListItem
  startIndex: number
  endIndex: number
  completed: number
  active: number
  total: number
}

export function buildRootSummaries(tasks: readonly TaskListItem[]): RootSummary[] {
  const rootStartIndexes: number[] = []
  for (let i = 0; i < tasks.length; i++) {
    const task = tasks[i]
    if (!task) continue
    if (task.depth !== 0) continue
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
      if (!rowTask) continue

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

export function findOwningRootIndex(tasks: readonly TaskListItem[], rowIndex: number): number | null {
  if (rowIndex < 0 || rowIndex >= tasks.length) return null

  for (let i = rowIndex; i >= 0; i--) {
    const task = tasks[i]
    if (!task) continue
    if (task.depth !== 0) continue
    return i
  }

  return null
}
