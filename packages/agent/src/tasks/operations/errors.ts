import {
  formatDuplicateTaskIdError,
  formatInvalidAssigneeError,
  formatInvalidTaskTypeError,
  formatMissingAssignmentMessageError,
  formatMissingAssignmentRoleError,
  formatTaskCompletionBlockedError,
  formatTaskNotFoundError,
  formatTaskParentNotFoundError,
} from '../../prompts/error-states'
import type { TaskTypeId } from '../registry'

export type TaskOperationErrorCode =
  | 'task_not_found'
  | 'parent_not_found'
  | 'duplicate_task_id'
  | 'invalid_task_type'
  | 'invalid_assignee'
  | 'missing_assignment_message'
  | 'missing_assignment_role'
  | 'completion_blocked'
  | 'invalid_status_transition'
  | 'empty_update_patch'
  | 'worker_not_found'

export interface TaskOperationErrorDetails {
  readonly code: TaskOperationErrorCode
  readonly message: string
}

export function taskNotFound(taskId: string): TaskOperationErrorDetails {
  return { code: 'task_not_found', message: formatTaskNotFoundError(taskId) }
}

export function parentNotFound(taskId: string, parentId: string): TaskOperationErrorDetails {
  return { code: 'parent_not_found', message: formatTaskParentNotFoundError(taskId, parentId) }
}

export function duplicateTaskId(taskId: string): TaskOperationErrorDetails {
  return { code: 'duplicate_task_id', message: formatDuplicateTaskIdError(taskId) }
}

export function invalidTaskType(taskId: string, taskType: TaskTypeId | string): TaskOperationErrorDetails {
  return { code: 'invalid_task_type', message: formatInvalidTaskTypeError(taskId, taskType) }
}

export function invalidAssignee(taskId: string, assignee: string): TaskOperationErrorDetails {
  return { code: 'invalid_assignee', message: formatInvalidAssigneeError(taskId, assignee) }
}

export function missingAssignmentMessage(taskId: string): TaskOperationErrorDetails {
  return { code: 'missing_assignment_message', message: formatMissingAssignmentMessageError(taskId) }
}

export function missingAssignmentRole(taskId: string): TaskOperationErrorDetails {
  return { code: 'missing_assignment_role', message: formatMissingAssignmentRoleError(taskId) }
}

export function completionBlocked(taskId: string): TaskOperationErrorDetails {
  return { code: 'completion_blocked', message: formatTaskCompletionBlockedError(taskId) }
}

export function invalidStatusTransition(taskId: string, from: string, to: string): TaskOperationErrorDetails {
  return {
    code: 'invalid_status_transition',
    message: `Task update rejected: cannot transition task "${taskId}" from "${from}" to "${to}".`,
  }
}

export function emptyUpdatePatch(taskId: string): TaskOperationErrorDetails {
  return {
    code: 'empty_update_patch',
    message: `Task update rejected: no changes provided for task "${taskId}".`,
  }
}

export function workerNotFound(taskId: string): TaskOperationErrorDetails {
  return {
    code: 'worker_not_found',
    message: `Worker operation rejected: task "${taskId}" has no active worker.`,
  }
}

export function toTurnErrorMessage(error: TaskOperationErrorDetails): string {
  return error.message
}
