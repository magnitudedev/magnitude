/**
 * ApprovalWorker
 *
 * Routes tool_approved/tool_rejected events from the CLI to the
 * ApprovalStateService, which resolves the Deferred that the
 * permission gate is blocking on.
 */

import { Effect } from 'effect'
import { Worker } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { ExecutionManager } from '../execution/execution-manager'

// =============================================================================
// Worker
// =============================================================================

export const ApprovalWorker = Worker.define<AppEvent>()({
  name: 'ApprovalWorker',

  eventHandlers: {

    tool_approved: (event, publish) => Effect.gen(function* () {
      const execManager = yield* ExecutionManager
      yield* execManager.approvalState.resolveApproval(event.toolCallId, 'approved')
    }),

    tool_rejected: (event, publish) => Effect.gen(function* () {
      const execManager = yield* ExecutionManager
      yield* execManager.approvalState.resolveApproval(event.toolCallId, 'rejected')
    })
  }
})
