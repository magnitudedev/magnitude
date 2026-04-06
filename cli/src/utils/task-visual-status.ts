import type { TaskListItem } from '../components/chat/types'

export type VisualStatus = 'completed' | 'pending'

const STATUS_PRIORITY: Record<VisualStatus, number> = {
  pending: 0,
  completed: -1, // completed never propagates
}

export function getOwnVisualStatus(task: TaskListItem): VisualStatus {
  if (task.status === 'completed') return 'completed'
  return 'pending'
}

export function computeInheritedVisualStatusMap(
  tasks: readonly TaskListItem[]
): Map<string, VisualStatus> {
  const result = new Map<string, VisualStatus>()

  // Forward pass: own status for all rows
  for (const task of tasks) {
    result.set(task.taskId, getOwnVisualStatus(task))
  }

  // Reverse pass: child -> parent inheritance
  for (let i = tasks.length - 1; i >= 0; i--) {
    const task = tasks[i]

    const childStatus = result.get(task.taskId)
    if (!childStatus) continue

    // Completed status never propagates upward
    if (childStatus === 'completed') continue

    // No parent to propagate to
    if (!task.parentId) continue

    const parentStatus = result.get(task.parentId)

    // Missing parent in visible list: safe no-op
    if (parentStatus === undefined) continue

    // Completed parents are terminal and never inherit
    if (parentStatus === 'completed') continue

    if (STATUS_PRIORITY[childStatus] > STATUS_PRIORITY[parentStatus]) {
      result.set(task.parentId, childStatus)
    }
  }

  return result
}
