/**
 * ReplayProjection (Forked)
 *
 * Tracks ReactorState per fork for xml-act crash recovery.
 * Reconstructs the state from tool_event AppEvents using xml-act's
 * foldReactorState() pure fold function — direct pass-through, no conversion.
 *
 * On session resume, the accumulated ReactorState is passed to
 * runtime.streamWith({ initialState }) so completed tools are skipped.
 */

import { Projection } from '@magnitudedev/event-core'
import { initialEngineState, foldEngineState } from '@magnitudedev/xml-act'
import type { EngineState } from '@magnitudedev/xml-act'
import type { AppEvent } from '../events'

export const ReplayProjection = Projection.defineForked<AppEvent, EngineState>()({
  name: 'Replay',

  initialFork: initialEngineState(),

  eventHandlers: {
    tool_event: ({ event, fork }) => {
      // ToolCallEvent is the inner union — forward relevant variants to foldReactorState
      switch (event.event._tag) {
        case 'ToolInputStarted':
        case 'ToolParseError':
        case 'StructuralParseError':
        case 'ToolExecutionEnded':
          return foldEngineState(fork, event.event)
        default:
          return fork
      }
    },

    // Keep state through turn_started so crash recovery can read it.
    // Reset on turn_outcome — terminal turns don't need replay.
    // Only a crashed turn (no turn_outcome) retains its state for recovery.
    turn_started: ({ fork }) => fork,
    turn_outcome: () => initialEngineState(),
  },
})
