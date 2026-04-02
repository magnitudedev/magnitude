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
        : { kind: 'lead' },
    workerForkId: task.worker?.forkId ?? null,
  }
}

export function flattenTaskTree(state: TaskGraphState): TaskListItem[] {
  const rows: TaskListItem[] = []

  const visit = (taskId: string, depth: number) => {
    const task = state.tasks.get(taskId)
    if (!task) return
    rows.push(toTaskListItem(task, depth))
    for (const childId of task.childIds) visit(childId, depth + 1)
  }

  for (const rootId of state.rootTaskIds) visit(rootId, 0)

  return rows
}
