/**
 * OutboundMessagesProjection
 *
 * Buffers streamed message_start/chunk/end events into completed outbound messages.
 */

import { Projection, Signal } from '@magnitudedev/event-core'
import { Schema } from 'effect'
import type { AppEvent } from '../events'
import { AgentRoutingProjection, getRoutingEntryByForkId } from './agent-routing'
import type { MessageDestination } from '../events'

export interface OutboundMessageCompletedSignal {
  readonly id: string
  readonly forkId: string | null
  readonly destination: MessageDestination
  readonly text: string
  readonly targetForkId: string | null | undefined
  readonly userFacing: boolean
  readonly timestamp: number
}

const MessageDestinationSchema = Schema.Union(
  Schema.Struct({ kind: Schema.Literal('user') }),
  Schema.Struct({ kind: Schema.Literal('coordinator') }),
  Schema.Struct({ kind: Schema.Literal('worker'), agentId: Schema.String }),
)

const PendingOutboundMessageSchema = Schema.Struct({
  forkId: Schema.NullOr(Schema.String),
  destination: MessageDestinationSchema,
  text: Schema.String,
})
export type PendingOutboundMessage = typeof PendingOutboundMessageSchema.Type

export const OutboundMessagesStateSchema = Schema.Struct({
  pendingMessages: Schema.ReadonlyMap({ key: Schema.String, value: PendingOutboundMessageSchema }),
})
export type OutboundMessagesState = typeof OutboundMessagesStateSchema.Type

export const OutboundMessagesProjection = Projection.define<AppEvent>()({
  name: 'OutboundMessages',
  state: OutboundMessagesStateSchema,
  reads: [AgentRoutingProjection] as const,

  initial: {
    pendingMessages: new Map<string, PendingOutboundMessage>(),
  },

  signals: {
    messageCompleted: Signal.create<OutboundMessageCompletedSignal>('OutboundMessages/messageCompleted'),
  },

  eventHandlers: {
    message_start: ({ event, state }) => {
      const pendingMessages = new Map(state.pendingMessages)
      pendingMessages.set(event.id, {
        forkId: event.forkId,
        destination: event.destination,
        text: '',
      })
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

      const agentState = read(AgentRoutingProjection)

      let targetForkId: string | null | undefined
      let userFacing: boolean

      switch (entry.destination.kind) {
        case 'user':
          targetForkId = null
          userFacing = true
          break
        case 'coordinator':
          targetForkId = entry.forkId !== null
            ? getRoutingEntryByForkId(agentState, entry.forkId)?.parentForkId ?? null
            : null
          userFacing = false
          break
        case 'worker': {
          targetForkId = agentState.agents.get(entry.destination.agentId)?.forkId
          userFacing = false
          break
        }
      }

      emit.messageCompleted({
        id: event.id,
        forkId: entry.forkId,
        destination: entry.destination,
        text: entry.text,
        targetForkId,
        userFacing,
        timestamp: event.timestamp,
      })

      return { ...state, pendingMessages }
    },
  },
})
