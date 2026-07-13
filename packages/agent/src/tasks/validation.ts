import { isRoleId } from '../agents/role-validation'
import type { TaskAssignee, WorkerAssignee } from './types'

function isWorkerAssignee(value: string): value is WorkerAssignee {
  return isRoleId(value) && value !== 'leader'
}

export function parseTaskAssignee(value: string): TaskAssignee | null {
  if (value === 'user') return 'user'
  if (isWorkerAssignee(value)) return value
  return null
}

export function assertTaskAssignee(value: string): TaskAssignee {
  const parsed = parseTaskAssignee(value)
  if (!parsed) throw new Error(`Invalid task assignee "${value}". Expected "user" or a valid worker role.`)
  return parsed
}
