/**
 * UserPresenceProjection
 *
 * Tracks terminal window focus state and emits presence signals
 * for the orchestrator when focus changes.
 */

import { Projection, Signal } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'

export interface UserPresenceState {
  readonly currentFocusState: boolean | null
  readonly blurredAt: number | null
  readonly focusedAt: number | null
}

export const UserPresenceProjection = Projection.define<AppEvent, UserPresenceState>()({
  name: 'UserPresence',

  initial: {
    currentFocusState: null,
    blurredAt: null,
    focusedAt: null,
  },

  signals: {
    presenceChanged: Signal.create<{ focused: boolean }>('UserPresence/presenceChanged'),
    userReturnedAfterAbsence: Signal.create<{}>('UserPresence/userReturnedAfterAbsence'),
  },

  eventHandlers: {
    window_focus_changed: ({ event, state, emit }) => {
      if (!event.focused) {
        return { ...state, currentFocusState: false, blurredAt: event.timestamp }
      }
      emit.presenceChanged({ focused: true })
      return { ...state, currentFocusState: true, focusedAt: event.timestamp }
    },

    user_return_confirmed: ({ state, emit }) => {
      emit.userReturnedAfterAbsence({})
      return state
    },
  },
})