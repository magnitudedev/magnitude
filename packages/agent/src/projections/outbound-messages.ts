/**
 * OutboundMessagesProjection
 *
 * Buffers streamed message_start/chunk/end events into completed outbound messages.
 */

import { Projection, Signal } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { ForkProjection } from './fork'
import { AgentRegistryProjection } from './agent-registry'

export interface PendingOutboundMessage {
  readonly forkId: string | null
  readonly dest: string
  readonly text: string
}

export interface OutboundMessagesState {
  readonly pendingMessages: ReadonlyMap<string, PendingOutboundMessage>
}

export interface OutboundMessageCompletedSignal {
  readonly id: string
  readonly forkId: string | null
  readonly dest: string
  readonly text: string
  readonly targetForkId: string | null | undefined
}

export const OutboundMessagesProjection = Projection.define<AppEvent, OutboundMessagesState>()({
  name: 'OutboundMessages',
  reads: [ForkProjection, AgentRegistryProjection] as const,

  initial: {
    pendingMessages: new Map(),
  },

  signals: {
    messageCompleted: Signal.create<OutboundMessageCompletedSignal>('OutboundMessages/messageCompleted'),
  },

  eventHandlers: {
    message_start: ({ event, state }) => {
      const pendingMessages = new Map(state.pendingMessages)
      pendingMessages.set(event.id, { forkId: event.forkId, dest: event.dest, text: '' })
      return { ...state, pendingMessages }
    },

    message_chunk: ({ event, state }) => {
      const entry = state.pendingMessages.get(event.id)
      if (!entry) return state

      const pendingMessages = new Map(state.pendingMessages)
      pendingMessages.set(event.id, { ...entry, text: entry.text + event.text })
      return { ...state, pendingMessages }
    },

    message_end: ({ event, state, emit, read }) => {
      const entry = state.pendingMessages.get(event.id)
      if (!entry) return state

      const pendingMessages = new Map(state.pendingMessages)
      pendingMessages.delete(event.id)

      const targetForkId = entry.dest === 'user'
        ? null
        : entry.dest === 'parent'
          ? entry.forkId === null
            ? null
            : read(ForkProjection).forks.get(entry.forkId)?.parentForkId
          : read(AgentRegistryProjection).agents.get(entry.dest)?.forkId

      emit.messageCompleted({
        id: event.id,
        forkId: entry.forkId,
        dest: entry.dest,
        text: entry.text,
        targetForkId,
      })

      return { ...state, pendingMessages }
    },
  },
})