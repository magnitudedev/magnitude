/**
 * ReplayProjection (Forked)
 *
 * Tracks ReactorState per fork for xml-act crash recovery (used by the
 * MockCortex/test-harness path which still drives ExecutionManager.execute
 * through xml-act's createTurnEngine).
 *
 * On session resume, the accumulated ReactorState is passed to
 * runtime.streamWith({ initialState }) so completed tools are skipped.
 *
 * NOTE: only the test-harness path consumes this. The native-paradigm live
 * runtime (Cortex worker) does not need it.
 */

import { Projection } from '@magnitudedev/event-core'
import { initialEngineState, foldEngineState } from '@magnitudedev/xml-act'
import type { EngineState } from '@magnitudedev/xml-act'
import type { AppEvent } from '../events'

export const ReplayProjection = Projection.defineForked<AppEvent, EngineState>()({
  name: 'Replay',

  initialFork: initialEngineState(),

  eventHandlers: {
    // Keep state through turn_started so crash recovery can read it.
    // Reset on turn_outcome — terminal turns don't need replay.
    // Only a crashed turn (no turn_outcome) retains its state for recovery.
    turn_started: ({ fork }) => fork,
    turn_outcome: () => initialEngineState(),
  },
})
