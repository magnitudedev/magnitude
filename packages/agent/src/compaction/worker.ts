/**
 * CompactionWorker — Orchestration layer for agentic compaction.
 *
 * Two-handler pattern:
 * 1. Signal handler (shouldCompactChanged): triggers compaction, runs agentic turn, emits events
 * 2. Event handler (turn_outcome): finalizes compaction when concurrent turn completes
 */

import { Effect } from 'effect'
import { Worker, AmbientServiceTag } from '@magnitudedev/event-core'
import { logger } from '@magnitudedev/logger'

import type { AppEvent } from '../events'

import { WindowProjection } from '../window'
import { AgentLifecycleProjection } from '../projections/agent-lifecycle'
import { CompactionProjection } from '../projections/compaction'
import { TurnProjection } from '../projections/turn'

import { getForkInfo } from '../agents/registry'
import { ConfigAmbient, getSlotConfigForRole } from '../ambient/config-ambient'

import { modelAttemptRetryability, presentModelAttemptFailure, type AgentModelAttemptFailure } from '../errors'
import { connectionRetrySchedule } from '../util/retry-backoff'

import { computeCompactionSizing } from './estimate'
import { runCompactionTurn, type CompactionTurnResult } from './turn'

// =============================================================================
// Retry predicate
// =============================================================================

/**
 * Predicate: should we retry this error?
 * Only retries retryable stream-start failures. Everything else fails immediately.
 */
function isRetryable(error: unknown): boolean {
  if (error && typeof error === 'object' && '_tag' in error) {
    return modelAttemptRetryability(error as AgentModelAttemptFailure)._tag === 'UpstreamRetryable'
  }
  return false
}

// =============================================================================
// Finalization
// =============================================================================

function injectCompaction(
  forkId: string | null,
  publish: (event: AppEvent) => Effect.Effect<void>,
): Effect.Effect<void> {
  return publish({
    type: 'compaction_injected',
    forkId,
  })
}

// =============================================================================
// Worker Definition
// =============================================================================

export const CompactionWorker = Worker.defineForked<AppEvent>()({
  name: 'CompactionWorker',

  forkLifecycle: {
    activateOn: 'agent_created',
    completeOn: ['agent_killed', 'worker_user_killed', 'worker_idle_closed'],
  },

  signalHandlers: (on) => [
    on(CompactionProjection.signals.shouldCompactChanged, (value, publish, read) =>
      Effect.gen(function* () {
        if (!value.shouldCompact) return
        const forkId = value.forkId
        const decline = (reason: string) => publish({
          type: 'compaction_declined',
          forkId,
          reason,
          ephemeral: true,
        })

        // 1. Guard: only compact from idle
        const compactionState = yield* read(CompactionProjection, forkId)
        if (!compactionState || compactionState._tag !== 'idle') {
          yield* decline('compaction state is not idle')
          return
        }

        // 2. Read window and config
        const windowState = yield* read(WindowProjection, forkId)
        if (!windowState || windowState.messages.length <= 1) {
          yield* decline('window has insufficient messages')
          return
        }

        const agentState = yield* read(AgentLifecycleProjection)
        const forkInfo = getForkInfo(agentState, forkId)
        if (!forkInfo) {
          yield* decline('fork metadata is unavailable')
          return
        }

        const { roleId } = forkInfo
        const ambientService = yield* AmbientServiceTag
        const configState = ambientService.getValue(ConfigAmbient)
        const roleConfig = getSlotConfigForRole(configState, roleId)

        // 3. Compute compaction sizing
        const { compactedMessageCount } = computeCompactionSizing(
          windowState.messages,
          roleConfig.softCap,
        )
        if (compactedMessageCount === 0) {
          yield* decline('compaction sizing selected no messages')
          return
        }

        // 4. Emit compaction_started (freezes count)
        yield* publish({
          type: 'compaction_started',
          forkId,
          compactedMessageCount,
        })

        // 5. Run agentic compaction turn with retry
        const result: CompactionTurnResult = yield* runCompactionTurn(forkId, roleId, windowState, roleConfig.softCap, publish, read, agentState).pipe(
          Effect.retry({
            schedule: connectionRetrySchedule,
            while: (err) => {
              if (!isRetryable(err)) return false
              logger.warn({ forkId, error: String(err) }, '[CompactionWorker] Retryable error, retrying')
              return true
            },
          }),
        )

        // 6. Emit compaction_prepared
        yield* publish({
          type: 'compaction_prepared',
          forkId,
          turn: result.turn,
          ...result.compactionOutcome,
          compactedMessageCount,
          inputTokens: result.inputTokens,
          outputTokens: result.outputTokens,
          refreshedContext: null,
        })

        // 7. Check if we can finalize immediately
        const turnState = yield* read(TurnProjection, forkId)
        if (!turnState || turnState._tag === 'idle' || turnState._tag === 'waiting_for_user') {
          yield* injectCompaction(forkId, publish)
        }
      }).pipe(
        Effect.onInterrupt(() => Effect.gen(function* () {
          logger.warn({ forkId: value.forkId }, '[CompactionWorker] Compaction interrupted')
          yield* publish({
            type: 'compaction_failed',
            forkId: value.forkId,
            error: 'Compaction interrupted',
            presentation: null,
          })
        }).pipe(Effect.orDie)),
        Effect.catchAll((error: unknown) => Effect.gen(function* () {
          // Present stream-start failures without going through TurnOutcome.
          let presentation = null
          if (error && typeof error === 'object' && '_tag' in error) {
            presentation = presentModelAttemptFailure(error as AgentModelAttemptFailure)
          }

          const message = error instanceof Error ? error.message : String(error)
          logger.error({ forkId: value.forkId, error: message }, '[CompactionWorker] Compaction failed')
          yield* publish({
            type: 'compaction_failed',
            forkId: value.forkId,
            error: message,
            presentation,
          })
        })),
      ),
    ),
  ],

  eventHandlers: {
    turn_outcome: (event, publish, read) =>
      Effect.gen(function* () {
        const forkId = event.forkId
        const compactionState = yield* read(CompactionProjection, forkId)
        if (!compactionState || compactionState._tag !== 'pendingInjection') return

        yield* injectCompaction(forkId, publish)
      }),
  },
})
