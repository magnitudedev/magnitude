/**
 * AgentRoutingProjection
 *
 * Global projection tracking child agent message routing.
 */

import { Projection, Signal } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'

export interface RoutingEntry {
  readonly agentId: string
  readonly forkId: string
  readonly parentForkId: string | null
}

export interface AgentRoutingState {
  readonly agents: ReadonlyMap<string, RoutingEntry>
  readonly agentByForkId: ReadonlyMap<string, string>
  readonly pendingMessages: ReadonlyMap<string, { forkId: string | null; dest: string; text: string }>
  readonly deferredParentMessages: ReadonlyMap<string, readonly { text: string }[]>
}

export function getRoutingEntry(state: AgentRoutingState, agentId: string): RoutingEntry | undefined {
  return state.agents.get(agentId)
}

export function getRoutingEntryByForkId(state: AgentRoutingState, forkId: string): RoutingEntry | undefined {
  const agentId = state.agentByForkId.get(forkId)
  if (!agentId) return undefined
  return state.agents.get(agentId)
}

export function isActiveRoute(state: AgentRoutingState, agentId: string): boolean {
  return state.agents.has(agentId)
}

export interface AgentMessageSignal {
  readonly targetForkId: string
  readonly agentId: string
  readonly message: string
  readonly timestamp: number
}

export interface AgentResponseSignal {
  readonly targetForkId: string | null
  readonly agentId: string
  readonly message: string
  readonly timestamp: number
}

export interface AgentCommunicationStreamSignal {
  readonly streamId: string
  readonly targetForkId: string
  readonly direction: 'from_agent' | 'to_agent'
  readonly agentId: string
  readonly textDelta: string
  readonly timestamp: number
}

export interface AgentCommunicationStreamCompletedSignal {
  readonly streamId: string
  readonly targetForkId: string
  readonly direction: 'from_agent' | 'to_agent'
  readonly agentId: string
  readonly timestamp: number
}

export const AgentRoutingProjection = Projection.define<AppEvent, AgentRoutingState>()(({
  name: 'AgentRouting',

  initial: {
    agents: new Map(),
    agentByForkId: new Map(),
    pendingMessages: new Map(),
    deferredParentMessages: new Map(),
  },

  signals: {
    agentRegistered: Signal.create<{ forkId: string; parentForkId: string | null }>('AgentRouting/registered'),
    agentMessage: Signal.create<AgentMessageSignal>('AgentRouting/message'),
    agentResponse: Signal.create<AgentResponseSignal>('AgentRouting/response'),
    communicationStreamStarted: Signal.create<AgentCommunicationStreamSignal>('AgentRouting/communicationStreamStarted'),
    communicationStreamChunk: Signal.create<AgentCommunicationStreamSignal>('AgentRouting/communicationStreamChunk'),
    communicationStreamCompleted: Signal.create<AgentCommunicationStreamCompletedSignal>('AgentRouting/communicationStreamCompleted'),
  },

  eventHandlers: {
    agent_created: ({ event, state, emit }) => {
      const existingAgent = state.agents.get(event.agentId)
      if (existingAgent) {
        throw new Error(`[AgentRoutingProjection] Invalid state transition: agent_created for already existing agent ${event.agentId} (forkId: ${existingAgent.forkId})`)
      }

      const existingForkAgentId = state.agentByForkId.get(event.forkId)
      if (existingForkAgentId) {
        throw new Error(`[AgentRoutingProjection] Invalid state transition: agent_created for already indexed fork ${event.forkId} (agentId: ${existingForkAgentId})`)
      }

      const entry: RoutingEntry = {
        agentId: event.agentId,
        forkId: event.forkId,
        parentForkId: event.parentForkId,
      }

      emit.agentRegistered({ forkId: event.forkId, parentForkId: event.parentForkId })

      return {
        ...state,
        agents: new Map(state.agents).set(event.agentId, entry),
        agentByForkId: new Map(state.agentByForkId).set(event.forkId, event.agentId),
      }
    },


    message_start: ({ event, state, emit }) => {
      const pendingMessages = new Map(state.pendingMessages)
      pendingMessages.set(event.id, { forkId: event.forkId, dest: event.dest, text: '' })

      if (event.dest !== 'user') {
        if (event.dest === 'parent' && event.forkId !== null) {
          const source = getRoutingEntryByForkId(state, event.forkId)
          if (source) {
            emit.communicationStreamStarted({
              streamId: event.id,
              targetForkId: source.forkId,
              direction: 'to_agent',
              agentId: source.agentId,
              textDelta: '',
              timestamp: event.timestamp,
            })
          }
        } else if (isActiveRoute(state, event.dest)) {
          const target = getRoutingEntry(state, event.dest)
          if (target) {
            emit.communicationStreamStarted({
              streamId: event.id,
              targetForkId: target.forkId,
              direction: 'from_agent',
              agentId: target.agentId,
              textDelta: '',
              timestamp: event.timestamp,
            })
          }
        }
      }

      return { ...state, pendingMessages }
    },

    message_chunk: ({ event, state, emit }) => {
      const entry = state.pendingMessages.get(event.id)
      if (!entry) return state

      const pendingMessages = new Map(state.pendingMessages)
      pendingMessages.set(event.id, { ...entry, text: entry.text + event.text })

      if (entry.dest !== 'user' && event.text.length > 0) {
        if (entry.dest === 'parent' && entry.forkId !== null) {
          const source = getRoutingEntryByForkId(state, entry.forkId)
          if (source) {
            emit.communicationStreamChunk({
              streamId: event.id,
              targetForkId: source.forkId,
              direction: 'to_agent',
              agentId: source.agentId,
              textDelta: event.text,
              timestamp: event.timestamp,
            })
          }
        } else if (isActiveRoute(state, entry.dest)) {
          const target = getRoutingEntry(state, entry.dest)
          if (target) {
            emit.communicationStreamChunk({
              streamId: event.id,
              targetForkId: target.forkId,
              direction: 'from_agent',
              agentId: target.agentId,
              textDelta: event.text,
              timestamp: event.timestamp,
            })
          }
        }
      }

      return { ...state, pendingMessages }
    },

    message_end: ({ event, state, emit }) => {
      const entry = state.pendingMessages.get(event.id)
      if (!entry) return state

      const pendingMessages = new Map(state.pendingMessages)
      pendingMessages.delete(event.id)

      let nextState: AgentRoutingState = { ...state, pendingMessages }

      if (entry.dest === 'parent' && entry.forkId !== null) {
        const source = getRoutingEntryByForkId(state, entry.forkId)
        if (source) {
          emit.communicationStreamCompleted({
            streamId: event.id,
            targetForkId: source.forkId,
            direction: 'to_agent',
            agentId: source.agentId,
            timestamp: event.timestamp,
          })
        }

        const existing = state.deferredParentMessages.get(entry.forkId) ?? []
        const deferredParentMessages = new Map(state.deferredParentMessages)
        deferredParentMessages.set(entry.forkId, [...existing, { text: entry.text }])
        nextState = { ...nextState, deferredParentMessages }
      }

      if (entry.dest !== 'user' && entry.dest !== 'parent' && isActiveRoute(state, entry.dest)) {
        const target = getRoutingEntry(state, entry.dest)
        if (target) {
          emit.communicationStreamCompleted({
            streamId: event.id,
            targetForkId: target.forkId,
            direction: 'from_agent',
            agentId: target.agentId,
            timestamp: event.timestamp,
          })

          emit.agentMessage({
            targetForkId: target.forkId,
            agentId: entry.dest,
            message: entry.text,
            timestamp: event.timestamp,
          })
        }
      }

      return nextState
    },

    turn_completed: ({ event, state, emit }) => {
      if (event.forkId === null) return state

      const messages = state.deferredParentMessages.get(event.forkId)
      if (!messages || messages.length === 0) return state

      const deferredParentMessages = new Map(state.deferredParentMessages)
      deferredParentMessages.delete(event.forkId)

      if (!event.result.success || event.result.turnDecision !== 'yield') {
        return { ...state, deferredParentMessages }
      }

      const agent = getRoutingEntryByForkId(state, event.forkId)
      if (!agent) {
        return { ...state, deferredParentMessages }
      }

      const fullText = messages.map(message => message.text).join('\n').trim()

      emit.agentResponse({
        targetForkId: agent.parentForkId,
        agentId: agent.agentId,
        message: fullText,
        timestamp: event.timestamp,
      })

      return { ...state, deferredParentMessages }
    },

    agent_killed: ({ event, state }) => {
      const routedAgentId = state.agentByForkId.get(event.forkId)
      if (!routedAgentId) return state
      if (routedAgentId !== event.agentId) return state

      const agents = new Map(state.agents)
      agents.delete(event.agentId)

      const agentByForkId = new Map(state.agentByForkId)
      agentByForkId.delete(event.forkId)

      const pendingMessages = new Map(state.pendingMessages)
      for (const [id, pending] of pendingMessages.entries()) {
        if (pending.forkId === event.forkId || pending.dest === event.agentId) {
          pendingMessages.delete(id)
        }
      }

      const deferredParentMessages = new Map(state.deferredParentMessages)
      deferredParentMessages.delete(event.forkId)

      return {
        ...state,
        agents,
        agentByForkId,
        pendingMessages,
        deferredParentMessages,
      }
    },
  },
}))