import type { TaskListItem } from '../components/chat/types'

type TaskRecord = {
  id: string
  title: string
  taskType: string
  parentId: string | null
  childIds: readonly string[]
  assignee: unknown
  worker: { agentId: string; forkId: string; role: string } | null
  status: 'pending' | 'working' | 'completed' | 'archived'
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

function makeArchivedSummaryRow(parentId: string, depth: number, archivedTasks: TaskRecord[]): TaskListItem {
  return {
    taskId: `__archived__${parentId}`,
    title: `${archivedTasks.length} archived tasks`,
    type: 'archived',
    status: 'archived',
    depth,
    parentId,
    createdAt: 0,
    updatedAt: 0,
    completedAt: null,
    assignee: { kind: 'none' },
    workerForkId: null,
  }
}

export function flattenTaskTree(state: TaskGraphState): TaskListItem[] {
  const rows: TaskListItem[] = []

  const visitChildren = (childIds: readonly string[], depth: number, parentId: string | null) => {
    const archivedTasks: TaskRecord[] = []
    const nonArchivedIds: string[] = []

    for (const childId of childIds) {
      const task = state.tasks.get(childId)
      if (!task) continue
      if (task.status === 'archived') archivedTasks.push(task)
      else nonArchivedIds.push(childId)
    }

    // Archived summary row first
    if (archivedTasks.length > 0) {
      rows.push(makeArchivedSummaryRow(parentId ?? '__root', depth, archivedTasks))
      // Include individual archived tasks as children of the summary
      for (const task of archivedTasks) {
        rows.push(toTaskListItem(task, depth + 1))
      }
    }

    // Then non-archived tasks
    for (const childId of nonArchivedIds) visit(childId, depth)
  }

  const visit = (taskId: string, depth: number) => {
    const task = state.tasks.get(taskId)
    if (!task) return
    rows.push(toTaskListItem(task, depth))
    visitChildren(task.childIds, depth + 1, task.id)
  }

  // Handle roots
  const archivedRoots: TaskRecord[] = []
  const nonArchivedRootIds: string[] = []
  for (const rootId of state.rootTaskIds) {
    const task = state.tasks.get(rootId)
    if (!task) continue
    if (task.status === 'archived') archivedRoots.push(task)
    else nonArchivedRootIds.push(rootId)
  }

  if (archivedRoots.length > 0) {
    rows.push(makeArchivedSummaryRow('__root', 0, archivedRoots))
    for (const task of archivedRoots) {
      rows.push(toTaskListItem(task, 1))
    }
  }

  for (const rootId of nonArchivedRootIds) visit(rootId, 0)

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

function isArchivedSummaryRow(task: TaskListItem): boolean {
  return task.taskId.startsWith('__archived__')
}

export function buildRootSummaries(tasks: readonly TaskListItem[]): RootSummary[] {
  const rootStartIndexes: number[] = []
  for (let i = 0; i < tasks.length; i++) {
    const task = tasks[i]
    if (!task) continue
    if (task.depth !== 0) continue
    if (isArchivedSummaryRow(task)) continue
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
      if (isArchivedSummaryRow(rowTask)) continue

      total += 1
      if (rowTask.status === 'completed' || rowTask.status === 'archived') {
        completed += 1
      } else if (rowTask.status === 'working') {
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
    if (isArchivedSummaryRow(task)) continue
    return i
  }

  return null
}
