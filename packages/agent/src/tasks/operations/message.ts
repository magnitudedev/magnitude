import { Effect } from 'effect'
import { TaskGraphStateReaderTag } from '../../tools/task-reader'
import { invalidTaskMessageRoute, taskMessageRouteFailed } from './errors'
import type { TaskDirectiveContext, TaskDirectiveErrorResult } from './handler'

export interface MessageDirective {
  readonly kind: 'message'
  readonly taskId: string | null
  readonly scope: 'top-level' | 'task'
  readonly defaultTopLevelDestination: 'user' | 'parent'
  readonly allowSingleUserReplyThisTurn: boolean
  readonly directUserRepliesSent: number
}

export type MessageDirectiveSuccess = {
  readonly success: true
  readonly destination: string
  readonly directUserRepliesSent: number
}

export const handleMessageDirective = (
  directive: MessageDirective,
  _context: TaskDirectiveContext,
) =>
  Effect.gen(function* () {
    if (directive.scope === 'top-level') {
      let destination = directive.defaultTopLevelDestination
      let sent = directive.directUserRepliesSent
      if (destination === 'user') {
        if (!directive.allowSingleUserReplyThisTurn || sent >= 1) {
          destination = 'parent'
        } else {
          sent += 1
        }
      }
      return { success: true, destination, directUserRepliesSent: sent } as const
    }

    if (!directive.taskId) {
      const err = invalidTaskMessageRoute('(unknown)')
      return { success: false, code: err.code, error: err.message } as const
    }

    const taskReader = yield* TaskGraphStateReaderTag
    const task = yield* taskReader.getTask(directive.taskId)
    if (!task) {
      const err = taskMessageRouteFailed(directive.taskId)
      return { success: false, code: err.code, error: err.message } as const
    }
    if (!task.worker?.agentId) {
      const err = invalidTaskMessageRoute(directive.taskId)
      return { success: false, code: err.code, error: err.message } as const
    }

    return {
      success: true,
      destination: task.worker.agentId,
      directUserRepliesSent: directive.directUserRepliesSent,
    } as const
  })
