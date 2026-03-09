/**
 * Approval State Service
 *
 * Manages the approval flow for gated tool calls.
 * Follows Sage's ApprovalState pattern exactly.
 *
 * Key responsibilities:
 * - Gates call requestApproval() which blocks until tool_approved/tool_rejected arrives
 * - Uses registered handler callbacks (not events/signals) to notify projections
 * - Handles hydration: if decision was already made (replayed), returns immediately
 *
 * Contract:
 * - Exactly one `pending` per approval (contains tool info for display)
 * - Exactly one `decided` per approval (contains decision)
 * - `pending` always emitted before `decided`
 */

import { Effect, Deferred, Context } from 'effect'
import type { ToolDisplay } from '../events'

// =============================================================================
// Types
// =============================================================================

export type ApprovalDecision = 'approved' | 'rejected'

export type ApprovalUpdate =
  | { readonly _tag: 'pending'; readonly toolCallId: string; readonly forkId: string | null; readonly toolKey: string; readonly input: unknown; readonly reason: string; readonly display?: ToolDisplay }
  | { readonly _tag: 'decided'; readonly toolCallId: string; readonly forkId: string | null; readonly decision: ApprovalDecision }

export type ApprovalHandler = (update: ApprovalUpdate) => void

// =============================================================================
// Internal Tracker
// =============================================================================

interface ApprovalTracker {
  toolKey: string
  input: unknown
  reason: string
  forkId: string | null
  display?: ToolDisplay
  decision: ApprovalDecision | null
  deferred: Deferred.Deferred<ApprovalDecision> | null
  emittedPending: boolean
  emittedDecided: boolean
}

function createTracker(forkId: string | null, toolKey: string, input: unknown, reason: string, display?: ToolDisplay): ApprovalTracker {
  return {
    toolKey,
    input,
    reason,
    forkId,
    display,
    decision: null,
    deferred: null,
    emittedPending: false,
    emittedDecided: false
  }
}

// =============================================================================
// Service Interface
// =============================================================================

export interface ApprovalStateService {
  /**
   * Request approval for a tool call. Called by the permission gate.
   *
   * 1. If decision was already made (replayed during hydration), returns immediately
   * 2. Otherwise, emits pending to handlers, creates Deferred, blocks
   * 3. When tool_approved/tool_rejected event arrives, resolves and returns decision
   */
  readonly requestApproval: (
    toolCallId: string,
    forkId: string | null,
    toolKey: string,
    input: unknown,
    reason: string,
    display?: ToolDisplay
  ) => Effect.Effect<ApprovalDecision>

  /**
   * Resolve an approval. Called by the approval worker when
   * tool_approved/tool_rejected events arrive.
   */
  readonly resolveApproval: (
    toolCallId: string,
    decision: ApprovalDecision
  ) => Effect.Effect<void>

  /**
   * Register a handler to receive approval updates.
   * DisplayProjection and WorkingState use this.
   */
  readonly registerHandler: (handler: ApprovalHandler) => void
}

export class ApprovalStateTag extends Context.Tag('ApprovalState')<
  ApprovalStateTag,
  ApprovalStateService
>() {}

// =============================================================================
// Implementation
// =============================================================================

/**
 * Create an ApprovalStateService instance.
 * Called from makeExecutionManager, scoped to manager lifetime.
 */
export function createApprovalState(): ApprovalStateService {
  const trackers = new Map<string, ApprovalTracker>()
  const handlers: ApprovalHandler[] = []

  function emitToHandlers(update: ApprovalUpdate): void {
    for (const handler of handlers) {
      handler(update)
    }
  }

  function tryEmitPending(tracker: ApprovalTracker, toolCallId: string): void {
    if (!tracker.emittedPending) {
      emitToHandlers({
        _tag: 'pending',
        toolCallId,
        forkId: tracker.forkId,
        toolKey: tracker.toolKey,
        input: tracker.input,
        reason: tracker.reason,
        ...(tracker.display ? { display: tracker.display } : {})
      })
      tracker.emittedPending = true
    }
  }

  function tryEmitDecided(tracker: ApprovalTracker, toolCallId: string): void {
    if (tracker.decision && tracker.emittedPending && !tracker.emittedDecided) {
      emitToHandlers({
        _tag: 'decided',
        toolCallId,
        forkId: tracker.forkId,
        decision: tracker.decision
      })
      tracker.emittedDecided = true
    }
  }

  return {
    requestApproval: (toolCallId, forkId, toolKey, input, reason, display) =>
      Effect.gen(function* () {
        let tracker = trackers.get(toolCallId)
        if (!tracker) {
          tracker = createTracker(forkId, toolKey, input, reason, display)
          trackers.set(toolCallId, tracker)
        } else {
          // Update tool info in case tracker was pre-created by resolveApproval during hydration
          tracker.toolKey = toolKey
          tracker.input = input
          tracker.reason = reason
          tracker.forkId = forkId
          tracker.display = display
        }

        // Hydration case — decision already arrived from replayed events
        if (tracker.decision) {
          tryEmitPending(tracker, toolCallId)
          tryEmitDecided(tracker, toolCallId)
          return tracker.decision
        }

        // Live case — emit pending, create deferred, block
        tryEmitPending(tracker, toolCallId)

        const deferred = yield* Deferred.make<ApprovalDecision>()
        tracker.deferred = deferred

        return yield* Deferred.await(deferred)
      }),

    resolveApproval: (toolCallId, decision) =>
      Effect.gen(function* () {
        let tracker = trackers.get(toolCallId)
        if (!tracker) {
          // Hydration: event arrived before requestApproval was called
          // Create a placeholder tracker — requestApproval will fill in tool info later
          tracker = createTracker(null, '', undefined, '')
          tracker.decision = decision
          trackers.set(toolCallId, tracker)
          return
        }

        tracker.decision = decision

        if (tracker.deferred) {
          // Live flow — requestApproval is waiting
          yield* Deferred.succeed(tracker.deferred, decision)
          tracker.deferred = null
          tryEmitDecided(tracker, toolCallId)
        } else {
          // Hydration — decision stored for when requestApproval is called later
          // If pending was already emitted, also emit decided
          tryEmitDecided(tracker, toolCallId)
        }
      }),

    registerHandler: (handler) => {
      handlers.push(handler)
    }
  }
}
