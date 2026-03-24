/**
 * ConversationProjection
 *
 * Tracks the clean user↔lead conversation for the root fork.
 * User messages come from user_message events.
 * Lead prose comes from message_* events filtered to dest='user'.
 * Turn boundaries flush accumulated prose into a conversation entry.
 *
 * Used by the reviewer agent to get injected with user intent context.
 */

import { Projection } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { textOf } from '../content'

// =============================================================================
// Types
// =============================================================================

export interface ConversationEntry {
  readonly role: 'user' | 'lead'
  readonly text: string
}

export interface ConversationState {
  readonly entries: readonly ConversationEntry[]
  /** Prose chunks accumulated during the current orchestrator turn */
  readonly pendingProse: string
  /** Message ids that target user and should be accumulated */
  readonly userMessageIds: ReadonlySet<string>
}

// =============================================================================
// Projection
// =============================================================================

export const ConversationProjection = Projection.define<AppEvent, ConversationState>()({
  name: 'Conversation',

  initial: {
    entries: [],
    pendingProse: '',
    userMessageIds: new Set(),
  },

  eventHandlers: {
    oneshot_task: ({ state }) => state,

    user_message: ({ event, state }) => {
      // Only track root fork (lead) conversation
      if (event.forkId !== null) return state

      const text = textOf(event.content)
      if (!text.trim()) return state

      return {
        ...state,
        entries: [...state.entries, { role: 'user' as const, text }],
      }
    },

    message_start: ({ event, state }) => {
      if (event.forkId !== null) return state
      if (event.dest !== 'user') return state
      return {
        ...state,
        userMessageIds: new Set(state.userMessageIds).add(event.id),
      }
    },

    message_chunk: ({ event, state }) => {
      // Only track root fork (lead) prose
      if (event.forkId !== null) return state
      if (!state.userMessageIds.has(event.id)) return state

      return {
        ...state,
        pendingProse: state.pendingProse + event.text,
      }
    },

    turn_completed: ({ event, state }) => {
      // Only track root fork
      if (event.forkId !== null) return state

      // Flush accumulated prose into an entry
      const prose = state.pendingProse.trim()
      if (!prose) {
        return { ...state, pendingProse: '', userMessageIds: new Set() }
      }

      return {
        entries: [...state.entries, { role: 'lead' as const, text: prose }],
        pendingProse: '',
        userMessageIds: new Set(),
      }
    },
  },
})
