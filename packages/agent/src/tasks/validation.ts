import { isValidVariant } from '../agents'
import type { TaskAssignee } from './types'
import { getTaskTypeDefinition, isTaskAssigneeAllowed, isValidTaskType, type TaskTypeId } from './registry'

export function parseTaskTypeId(value: string): TaskTypeId | null {
  return isValidTaskType(value) ? value : null
}

export function assertTaskTypeId(value: string): TaskTypeId {
  const parsed = parseTaskTypeId(value)
  if (!parsed) throw new Error(`Invalid task type "${value}".`)
  return parsed
}

export function parseTaskAssignee(value: string): TaskAssignee | null {
  if (value === 'self') return 'self'
  if (isValidVariant(value)) return value
  return null
}

export function assertTaskAssignee(value: string): TaskAssignee {
  const parsed = parseTaskAssignee(value)
  if (!parsed) throw new Error(`Invalid task assignee "${value}". Expected "self" or a valid worker variant.`)
  return parsed
}

export function validateTaskAssigneeForType(taskType: TaskTypeId, assignee: TaskAssignee): void {
  if (isTaskAssigneeAllowed(taskType, assignee)) return
  const allowed = getTaskTypeDefinition(taskType).allowedAssignees.join(', ')
  throw new Error(`Assignee "${assignee}" is not allowed for task type "${taskType}". Allowed: ${allowed}.`)
}
