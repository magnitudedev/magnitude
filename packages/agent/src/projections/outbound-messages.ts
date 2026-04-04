/**
 * OutboundMessagesProjection
 *
 * Buffers streamed message_start/chunk/end events into completed outbound messages.
 */

import { Projection, Signal } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { AgentRoutingProjection, getRoutingEntryByForkId } from './agent-routing'
import { TaskGraphProjection } from './task-graph'
import type { MessageDestination } from '../events'

export interface PendingOutboundMessage {
  readonly forkId: string | null
  readonly destination: MessageDestination
  readonly text: string
}

export interface OutboundMessagesState {
  readonly pendingMessages: ReadonlyMap<string, PendingOutboundMessage>
}

export interface OutboundMessageCompletedSignal {
  readonly id: string
  readonly forkId: string | null
  readonly destination: MessageDestination
  readonly text: string
  readonly targetForkId: string | null | undefined
  readonly userFacing: boolean
  readonly timestamp: number
}

export const OutboundMessagesProjection = Projection.define<AppEvent, OutboundMessagesState>()({
  name: 'OutboundMessages',
  reads: [AgentRoutingProjection, TaskGraphProjection] as const,

  initial: {
    pendingMessages: new Map(),
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
      const taskGraph = read(TaskGraphProjection)

      let targetForkId: string | null | undefined
      let userFacing: boolean

      switch (entry.destination.kind) {
        case 'user':
          targetForkId = null
          userFacing = true
          break
        case 'parent':
          targetForkId = entry.forkId !== null
            ? getRoutingEntryByForkId(agentState, entry.forkId)?.parentForkId ?? null
            : null
          userFacing = false
          break
        case 'worker': {
          const workerAgentId = taskGraph.tasks.get(entry.destination.taskId)?.worker?.agentId
          targetForkId = workerAgentId ? agentState.agents.get(workerAgentId)?.forkId : undefined
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