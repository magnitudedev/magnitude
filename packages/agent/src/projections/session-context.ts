/**
 * SessionContextProjection
 *
 * Holds session-level configuration that doesn't change after initialization.
 * Other projections can read from this via the `reads` mechanism.
 */

import { Projection } from '@magnitudedev/event-core'
import type { AppEvent, SessionContext as EventSessionContext } from '../events'

// =============================================================================
// Types
// =============================================================================

export interface SessionContextState {
  readonly initialized: boolean
  readonly context: EventSessionContext | null
}

// =============================================================================
// Projection
// =============================================================================

export const SessionContextProjection = Projection.define<AppEvent, SessionContextState>()({
  name: 'SessionContext',
  initial: { initialized: false, context: null },

  eventHandlers: {
    session_initialized: ({ event }) => ({
      initialized: true,
      context: event.context
    }),

    compaction_completed: ({ event, state }) => {
      // Update session context if refreshed during compaction
      if (event.refreshedContext) {
        return {
          ...state,
          context: event.refreshedContext
        }
      }
      return state
    }
  }
})
