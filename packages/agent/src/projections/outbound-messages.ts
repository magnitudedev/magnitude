/**
 * OutboundMessagesProjection
 *
 * Buffers streamed message_start/chunk/end events into completed outbound messages.
 */

import { Projection, Signal } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { AgentRoutingProjection, getRoutingEntryByForkId, type MessageScope } from './agent-routing'
import { TaskGraphProjection } from './task-graph'

export interface PendingOutboundMessage {
  readonly forkId: string | null
  readonly scope: MessageScope
  readonly taskId: string | null
  readonly text: string
}

export interface OutboundMessagesState {
  readonly pendingMessages: ReadonlyMap<string, PendingOutboundMessage>
}

export interface OutboundMessageCompletedSignal {
  readonly id: string
  readonly forkId: string | null
  readonly scope: MessageScope
  readonly taskId: string | null
  readonly text: string
  readonly targetForkId: string | null | undefined
  readonly userFacing: boolean
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
        scope: event.taskId !== null ? 'task' : 'top-level',
        taskId: event.taskId,
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

      const targetForkId = entry.scope === 'top-level'
        ? entry.forkId === null
          ? null
          : getRoutingEntryByForkId(agentState, entry.forkId)?.parentForkId
        : (() => {
            const workerAgentId = entry.taskId ? taskGraph.tasks.get(entry.taskId)?.worker?.agentId : undefined
            return workerAgentId ? agentState.agents.get(workerAgentId)?.forkId : undefined
          })()

      emit.messageCompleted({
        id: event.id,
        forkId: entry.forkId,
        scope: entry.scope,
        taskId: entry.taskId,
        text: entry.text,
        targetForkId,
        userFacing: entry.scope === 'top-level' && entry.forkId === null,
      })

      return { ...state, pendingMessages }
    },
  },
})