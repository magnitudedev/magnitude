import { Effect } from 'effect'
import type { TaskDirectiveContext } from './handler'

export interface MessageDirective {
  readonly kind: 'message'
  readonly defaultTopLevelDestination: 'user' | 'parent'
  readonly triggeredByUser: boolean
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
    let destination = directive.defaultTopLevelDestination
    let sent = directive.directUserRepliesSent
    if (destination === 'user') {
      if (!directive.triggeredByUser || sent >= 1) {
        destination = 'parent'
      } else {
        sent += 1
      }
    }
    return { success: true, destination, directUserRepliesSent: sent } as const
  })
