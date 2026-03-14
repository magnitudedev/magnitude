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

    agent_dismissed: ({ event, state }) => {
      const resolvedAgentId = state.agents.has(event.agentId)
        ? event.agentId
        : state.agentByForkId.get(event.forkId)

      if (!resolvedAgentId) {
        throw new Error(`[AgentRoutingProjection] Invalid state transition: agent_dismissed for unknown agent ${event.agentId} (forkId: ${event.forkId})`)
      }

      const existing = state.agents.get(resolvedAgentId)
      if (!existing) {
        throw new Error(`[AgentRoutingProjection] Invalid state transition: agent_dismissed for missing indexed agent ${resolvedAgentId}`)
      }

      const agents = new Map(state.agents)
      agents.delete(resolvedAgentId)
      const agentByForkId = new Map(state.agentByForkId)
      agentByForkId.delete(existing.forkId)

      return { ...state, agents, agentByForkId }
    },

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

    message_end: ({ event, state, emit }) => {
      const entry = state.pendingMessages.get(event.id)
      if (!entry) return state

      const pendingMessages = new Map(state.pendingMessages)
      pendingMessages.delete(event.id)

      let nextState: AgentRoutingState = { ...state, pendingMessages }

      if (entry.dest === 'parent' && entry.forkId !== null) {
        const existing = state.deferredParentMessages.get(entry.forkId) ?? []
        const deferredParentMessages = new Map(state.deferredParentMessages)
        deferredParentMessages.set(entry.forkId, [...existing, { text: entry.text }])
        nextState = { ...nextState, deferredParentMessages }
      }

      if (entry.dest !== 'user' && entry.dest !== 'parent' && isActiveRoute(state, entry.dest)) {
        const target = getRoutingEntry(state, entry.dest)
        if (target) {
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
  },
}))