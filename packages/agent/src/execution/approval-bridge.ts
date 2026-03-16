/**
 * Approval Bridge
 *
 * Bridges ApprovalState callbacks into DisplayProjection and WorkingStateProjection state.
 *
 * This runs during client initialization (before hydration) to ensure approval
 * cards appear correctly during both live execution and session replay.
 *
 * The bridge registers handlers with ApprovalState that update projection state
 * when approvals are requested or decided. This is the Sage pattern: ApprovalState
 * owns the coordination (deferreds, blocking), and the bridge pushes updates
 * into the display layer.
 */

import { Effect, SubscriptionRef } from 'effect'
import { ExecutionManager } from './execution-manager'
import { DisplayProjection, type ApprovalRequestMessage } from '../projections/display'
import { WorkingStateProjection } from '../projections/working-state'

/**
 * Register approval state handlers that bridge into projection state.
 * Must be called during client initialization, before hydration.
 */
export const registerApprovalBridge = Effect.gen(function* () {
  const executionManager = yield* ExecutionManager
  const displayProjection = yield* DisplayProjection.Tag
  const workingStateProjection = yield* WorkingStateProjection.Tag

  // Display: insert/update ApprovalRequestMessage in the fork's messages
  executionManager.approvalState.registerHandler((update) => {
    if (update._tag === 'pending') {
      const message: ApprovalRequestMessage = {
        id: update.toolCallId,
        type: 'approval_request',
        toolCallId: update.toolCallId,
        toolKey: update.toolKey,
        input: update.input,
        reason: update.reason,
        status: 'pending',
        timestamp: Date.now(),
        ...(update.display ? { display: update.display } : {})
      }
      Effect.runSync(SubscriptionRef.update(displayProjection.state, (state) => {
        const forkId = update.forkId
        const forkState = state.forks.get(forkId) ?? {
          status: 'idle' as const,
          messages: [],
          currentTurnId: null,
          streamingMessageId: null,
          activeThinkBlockId: null,
          showButton: 'send' as const,
          colorAssignments: new Map() as ReadonlyMap<string, number>,
        }
        const newForks = new Map(state.forks)
        newForks.set(forkId, {
          ...forkState,
          messages: [...forkState.messages, message]
        })
        return { ...state, forks: newForks }
      }))
    } else if (update._tag === 'decided') {
      Effect.runSync(SubscriptionRef.update(displayProjection.state, (state) => {
        const forkId = update.forkId
        const forkState = state.forks.get(forkId)
        if (!forkState) return state
        const newForks = new Map(state.forks)
        newForks.set(forkId, {
          ...forkState,
          messages: forkState.messages.map(m =>
            m.type === 'approval_request' && m.toolCallId === update.toolCallId
              ? { ...m, status: update.decision === 'approved' ? 'approved' as const : 'rejected' as const }
              : m
          )
        })
        return { ...state, forks: newForks }
      }))
    }
  })

  // WorkingState: set pendingApproval flag for the relevant fork
  executionManager.approvalState.registerHandler((update) => {
    const forkId = update.forkId
    const pending = update._tag === 'pending'
    Effect.runSync(SubscriptionRef.update(workingStateProjection.state, (state) => {
      const forkState = state.forks.get(forkId)
      if (!forkState) return state
      const newForks = new Map(state.forks)
      newForks.set(forkId, {
        ...forkState,
        pendingApproval: pending
      })
      return { ...state, forks: newForks }
    }))
  })
})
