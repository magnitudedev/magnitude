import { isValidVariant } from '../agents/variants'
import type { TaskAssignee, WorkerAssignee } from './types'

function isWorkerAssignee(value: string): value is WorkerAssignee {
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
