/**
 * ForkOrchestrator Worker
 *
 * Handles fork infrastructure that tools can't own:
 * - Root fork init on session start
 * - Disposing fork layers + browser harness on fork completion
 * - Auto-completing interrupted forks
 * - Scheduling delayed fork removal
 *
 * fork(), forkSync(), submit(), and task.validate handle their own lifecycle directly.
 */

import { Effect } from 'effect'
import { Worker } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { ExecutionManager } from '../execution/execution-manager'
import { ForkProjection } from '../projections/fork'
import { WorkingStateProjection } from '../projections/working-state'

import { BrowserService } from '../services/browser-service'
import { buildInterruptedTurnCompleted } from '../util/interrupt-utils'

// =============================================================================
// Worker
// =============================================================================

export const ForkOrchestrator = Worker.define<AppEvent>()({
  name: 'ForkOrchestrator',

  // interrupt handler must not be interrupted itself (it auto-completes interrupted forks)
  ignoreInterrupt: ['interrupt'] as const,

  eventHandlers: {

    // Create root fork resources when session starts
    session_initialized: (_event, _publish) => Effect.gen(function* () {
      const execManager = yield* ExecutionManager
      yield* execManager.initFork(null, 'orchestrator')
    }).pipe(Effect.orDie),

    // Auto-complete interrupted forks
    interrupt: (event, publish, read) => Effect.gen(function* () {
      const { forkId } = event
      if (forkId === null) return

      const forkState = yield* read(ForkProjection)
      const fork = forkState.forks.get(forkId)
      if (!fork || fork.status !== 'running') return

      yield* publish({
        type: 'fork_completed',
        forkId,
        parentForkId: fork.parentForkId,
        result: { interrupted: true },
      })
    }).pipe(Effect.orDie),

    // Dispose fork resources and schedule removal
    fork_completed: (event, publish) => Effect.gen(function* () {
      const browserService = yield* BrowserService
      yield* browserService.release(event.forkId)

      const execManager = yield* ExecutionManager
      yield* execManager.disposeFork(event.forkId)

      yield* Effect.forkDaemon(
        Effect.gen(function* () {
          yield* Effect.sleep("2 seconds")
          yield* publish({
            type: 'fork_removed',
            forkId: event.forkId,
            parentForkId: event.parentForkId,
          })
        })
      )
    }).pipe(Effect.orDie)
  },

  signalHandlers: (on) => [
    on(WorkingStateProjection.signals.turnInterrupted, ({ forkId, turnId, chainId }, publish) => Effect.gen(function* () {
      const turnCompleted = yield* buildInterruptedTurnCompleted({ forkId, turnId, chainId })
      yield* publish(turnCompleted)
    }))
  ]
})
