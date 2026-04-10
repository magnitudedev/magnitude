import { isValidVariant, type AgentVariant } from '../agents'
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

function isWorkerAssignee(value: string): value is Exclude<AgentVariant, 'lead' | 'lead-oneshot'> {
  return isValidVariant(value) && value !== 'lead' && value !== 'lead-oneshot'
}

export function parseTaskAssignee(value: string): TaskAssignee | null {
  if (value === 'user') return 'user'
  if (isWorkerAssignee(value)) return value
  return null
}

export function assertTaskAssignee(value: string): TaskAssignee {
  const parsed = parseTaskAssignee(value)
  if (!parsed) throw new Error(`Invalid task assignee "${value}". Expected "user" or a valid worker variant.`)
  return parsed
}

export function validateTaskAssigneeForType(taskType: TaskTypeId, assignee: TaskAssignee): void {
  if (isTaskAssigneeAllowed(taskType, assignee)) return

  const def = getTaskTypeDefinition(taskType)

  if (def.kind === 'composite') {
    throw new Error(`Task type "${taskType}" is a coordinator-only task and cannot be directly assigned.`)
  }

  if (def.kind === 'user') {
    throw new Error(`Task type "${taskType}" can only be assigned to the user.`)
  }

  const allowed = def.allowedAssignees.join(', ')
  throw new Error(`Assignee "${assignee}" is not allowed for task type "${taskType}". Allowed: ${allowed}.`)
}
