import type { TaskStatus } from '../../projections/task-graph'
import type { TaskOperationGraphSnapshot, TaskOperationTaskSnapshot } from './types'

const TASK_STATUSES = ['pending', 'working', 'completed', 'archived'] as const

export function isValidStatus(value: string): value is TaskStatus {
  return TASK_STATUSES.includes(value as TaskStatus)
}

export function canTransition(from: TaskStatus, to: TaskStatus): boolean {
  if (from === to) return false
  if (to === 'working') return false
  if (to === 'archived') return from === 'completed'
  if (to === 'pending') return from === 'completed' || from === 'archived'
  if (to === 'completed') return true
  return false
}

export function hasIncompleteChildren(taskId: string, graph: TaskOperationGraphSnapshot): boolean {
  const task = graph.tasks.get(taskId)
  if (!task) return false
  return task.childIds.some((childId) => {
    const child = graph.tasks.get(childId)
    if (!child) return false
    return child.status !== 'completed' && child.status !== 'archived'
  })
}

export function canCompleteTask(task: TaskOperationTaskSnapshot, childStatuses: readonly TaskStatus[]): boolean {
  if (task.status === 'completed') return false
  return childStatuses.every((status) => status === 'completed' || status === 'archived')
}
