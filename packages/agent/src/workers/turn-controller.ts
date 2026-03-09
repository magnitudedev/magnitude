/**
 * TurnController Worker
 *
 * Watches shouldTrigger signal and publishes turn_started events.
 * Now fork-aware - triggers turns for any fork.
 */

import { Effect } from 'effect'
import { Worker } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { WorkingStateProjection } from '../projections/working-state'
import { createId } from '../util/id'

// =============================================================================
// Worker
// =============================================================================

export const TurnController = Worker.define<AppEvent>()({
  name: 'TurnController',

  signalHandlers: (on) => [
    on(WorkingStateProjection.signals.shouldTriggerChanged, ({ forkId, shouldTrigger, chainId }, publish) => Effect.gen(function* () {
      // Only proceed if shouldTrigger is true
      if (!shouldTrigger) return

      // Generate IDs for new turn
      const turnId = createId()
      const newChainId = chainId ?? createId()

      // Publish turn_started for this fork
      yield* publish({
        type: 'turn_started',
        forkId,
        turnId,
        chainId: newChainId
      })
    }))
  ]
})
