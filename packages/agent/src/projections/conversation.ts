/**
 * ConversationProjection
 *
 * Tracks the clean user↔lead conversation for the root fork.
 * User messages come from user_message events.
 * Lead prose comes from top-level message_* events.
 * Turn boundaries flush accumulated prose into a conversation entry.
 *
 * Used by the reviewer agent to get injected with user intent context.
 */

import { Projection } from '@magnitudedev/event-core'
import { Schema } from 'effect'
import type { AppEvent } from '../events'
import { textOf } from '../content'
// textOf already uses UserPart internally
import { UserMessageResolutionProjection } from './user-message-resolution'

export const ConversationEntrySchema = Schema.Struct({
  role: Schema.Literal('user', 'lead'),
  text: Schema.String,
})
export type ConversationEntry = typeof ConversationEntrySchema.Type

export const ConversationStateSchema = Schema.Struct({
  entries: Schema.Array(ConversationEntrySchema),
  pendingProse: Schema.String,
  userMessageIds: Schema.ReadonlySet(Schema.String),
})
export type ConversationState = typeof ConversationStateSchema.Type

// =============================================================================
// Projection
// =============================================================================

export const ConversationProjection = Projection.define<AppEvent>()({
  name: 'Conversation',
  state: ConversationStateSchema,

  reads: [UserMessageResolutionProjection] as const,

  initial: {
    entries: [],
    pendingProse: '',
    userMessageIds: new Set<string>(),
  },

  eventHandlers: {
    message_start: ({ event, state }) => {
      if (event.forkId !== null) return state
      if (event.destination.kind !== 'user') return state
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

    turn_outcome: ({ event, state }) => {
      // Only track root fork
      if (event.forkId !== null) return state

      // Flush accumulated prose into an entry
      const prose = state.pendingProse.trim()
      if (!prose) {
        return { ...state, pendingProse: '', userMessageIds: new Set<string>() }
      }

      return {
        entries: [...state.entries, { role: 'lead' as const, text: prose }],
        pendingProse: '',
        userMessageIds: new Set<string>(),
      }
    },
  },

  signalHandlers: on => [
    on(UserMessageResolutionProjection.signals.userMessageResolved, ({ value, state }) => {
      if (value.forkId !== null) return state

      const text = textOf(value.content)
      if (!text.trim()) return state

      return {
        ...state,
        entries: [...state.entries, { role: 'user', text }],
      }
    }),
  ],
})
