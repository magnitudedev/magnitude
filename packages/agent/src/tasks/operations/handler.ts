import { Effect } from 'effect'
import type { TaskOperationGraphSnapshot } from './types'
import { handleCreateDirective, type CreateDirective } from './create'
import { handleUpdateDirective, type UpdateDirective } from './update'
import { handleAssignDirective, type AssignDirective } from './assign'
import { handleCancelDirective, type CancelDirective } from './cancel'
import { handleMessageDirective, type MessageDirective, type MessageDirectiveSuccess } from './message'

export interface TaskDirectiveContext {
  readonly forkId: string | null
  readonly timestamp: number
  readonly graph: TaskOperationGraphSnapshot
}

export type TaskDirectiveResult =
  | { success: true }
  | { success: false; code: string; error: string }

export type TaskDirectiveErrorResult = { success: false; code: string; error: string }

export type TaskDirective =
  | CreateDirective
  | UpdateDirective
  | AssignDirective
  | CancelDirective
  | MessageDirective

export type HandleTaskDirectiveResult = TaskDirectiveErrorResult | { success: true } | MessageDirectiveSuccess

export const handleTaskDirective = (
  directive: TaskDirective,
  context: TaskDirectiveContext,
) => {
  switch (directive.kind) {
    case 'create':
      return handleCreateDirective(directive, context)
    case 'update':
      return handleUpdateDirective(directive, context)
    case 'assign':
      return handleAssignDirective(directive, context)
    case 'cancel':
      return handleCancelDirective(directive, context)
    case 'message':
      return handleMessageDirective(directive, context)
  }
}
