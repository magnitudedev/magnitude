/**
 * ReplayProjection (Forked)
 *
 * Tracks EngineState per fork for crash recovery. On session resume,
 * the accumulated EngineState is passed to TurnEngine.runTurn({ initialState })
 * so completed tools are skipped and not re-executed.
 *
 * Folds tool_event AppEvents through the native engine state folder.
 * Resets on turn_outcome — only a crashed turn (no turn_outcome) retains
 * its state for recovery.
 */

import { Projection } from '@magnitudedev/event-core'
import { initialEngineState, foldEngineState } from '@magnitudedev/turn-engine'
import type { EngineState } from '@magnitudedev/turn-engine'
import type { AppEvent } from '../events'

export const ReplayProjection = Projection.defineForked<AppEvent, EngineState>()({
  name: 'Replay',

  initialFork: initialEngineState(),

  eventHandlers: {
    tool_event: ({ fork, event }) => foldEngineState(fork, event.event),

    // Keep state through turn_started so crash recovery can read it.
    turn_started: ({ fork }) => fork,

    // Reset on turn_outcome — terminal turns don't need replay.
    turn_outcome: () => initialEngineState(),
  },
})
