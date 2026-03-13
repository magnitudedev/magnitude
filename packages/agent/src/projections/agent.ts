/**
 * AgentProjection
 *
 * Global projection tracking child agent lifecycle and message routing.
 * This replaces both ForkProjection and AgentRegistryProjection.
 */

import { Projection, Signal } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'

// =============================================================================
// Types
// =============================================================================

export type AgentLifecycleStatus = 'running' | 'paused' | 'dismissed'

export interface AgentInstance {
  readonly agentId: string
  readonly forkId: string
  readonly parentForkId: string | null
  readonly name: string
  readonly role: string
  readonly context: string
  readonly mode: 'clone' | 'spawn'
  readonly taskId: string
  readonly message: string | null
  readonly outputSchema?: unknown
  readonly status: AgentLifecycleStatus
  readonly result?: unknown
  readonly dismissReason?: 'dismissed' | 'interrupted' | 'completed'
  readonly createdAt: number
  readonly dismissedAt?: number
}

export interface AgentState {
  readonly agents: ReadonlyMap<string, AgentInstance>
  readonly agentByForkId: ReadonlyMap<string, string>
  readonly pendingMessages: ReadonlyMap<string, { forkId: string | null; dest: string; text: string }>
  readonly deferredParentMessages: ReadonlyMap<string, readonly { text: string }[]>
}

// =============================================================================
// Helpers
// =============================================================================

export function getAgentByForkId(state: AgentState, forkId: string): AgentInstance | undefined {
  const agentId = state.agentByForkId.get(forkId)
  if (!agentId) return undefined
  return state.agents.get(agentId)
}

export function getActiveAgent(state: AgentState, agentId: string): AgentInstance | undefined {
  const agent = state.agents.get(agentId)
  if (!agent || agent.status === 'dismissed') return undefined
  return agent
}

// =============================================================================
// Signal Types
// =============================================================================

export interface AgentCreatedSignal {
  readonly forkId: string
  readonly parentForkId: string | null
  readonly agentId: string
  readonly name: string
  readonly role: string
  readonly taskId: string
  readonly mode: 'clone' | 'spawn'
}

export interface AgentDismissedSignal {
  readonly forkId: string
  readonly parentForkId: string | null
  readonly agentId: string
  readonly name: string
  readonly role: string
  readonly taskId: string
  readonly result: unknown
  readonly reason: 'dismissed' | 'interrupted' | 'completed'
}

export interface AgentPausedSignal {
  readonly forkId: string
  readonly parentForkId: string | null
  readonly agentId: string
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

// =============================================================================
// Projection
// =============================================================================

export const AgentProjection = Projection.define<AppEvent, AgentState>()(({
  name: 'Agent',

  initial: {
    agents: new Map(),
    agentByForkId: new Map(),
    pendingMessages: new Map(),
    deferredParentMessages: new Map(),
  },

  signals: {
    agentCreated: Signal.create<AgentCreatedSignal>('Agent/created'),
    agentDismissed: Signal.create<AgentDismissedSignal>('Agent/dismissed'),
    agentPaused: Signal.create<AgentPausedSignal>('Agent/paused'),
    agentMessage: Signal.create<AgentMessageSignal>('Agent/message'),
    agentResponse: Signal.create<AgentResponseSignal>('Agent/response'),
  },

  eventHandlers: {
    user_message: ({ state }) => state,

    agent_created: ({ event, state, emit }) => {
      const existingAgent = state.agents.get(event.agentId)
      if (existingAgent) {
        throw new Error(`[AgentProjection] Invalid state transition: agent_created for already existing agent ${event.agentId} (forkId: ${existingAgent.forkId})`)
      }

      const existingForkAgentId = state.agentByForkId.get(event.forkId)
      if (existingForkAgentId) {
        throw new Error(`[AgentProjection] Invalid state transition: agent_created for already indexed fork ${event.forkId} (agentId: ${existingForkAgentId})`)
      }

      emit.agentCreated({
        forkId: event.forkId,
        parentForkId: event.parentForkId,
        agentId: event.agentId,
        name: event.name,
        role: event.role,
        taskId: event.taskId,
        mode: event.mode,
      })

      const instance: AgentInstance = {
        agentId: event.agentId,
        forkId: event.forkId,
        parentForkId: event.parentForkId,
        name: event.name,
        role: event.role,
        context: event.context,
        mode: event.mode,
        taskId: event.taskId,
        message: event.message ?? null,
        outputSchema: event.outputSchema,
        status: 'running',
        createdAt: event.timestamp,
      }

      return {
        ...state,
        agents: new Map(state.agents).set(event.agentId, instance),
        agentByForkId: new Map(state.agentByForkId).set(event.forkId, event.agentId),
      }
    },

    agent_paused: ({ event, state, emit }) => {
      const existing = state.agents.get(event.agentId)
      if (!existing || existing.status === 'dismissed') return state

      emit.agentPaused({
        forkId: existing.forkId,
        parentForkId: existing.parentForkId,
        agentId: existing.agentId,
      })

      return {
        ...state,
        agents: new Map(state.agents).set(event.agentId, {
          ...existing,
          status: 'paused',
        }),
      }
    },

    agent_dismissed: ({ event, state, emit }) => {
      const resolvedAgentId = state.agents.has(event.agentId)
        ? event.agentId
        : state.agentByForkId.get(event.forkId)

      if (!resolvedAgentId) {
        throw new Error(`[AgentProjection] Invalid state transition: agent_dismissed for unknown agent ${event.agentId} (forkId: ${event.forkId})`)
      }

      const existing = state.agents.get(resolvedAgentId)
      if (!existing) {
        throw new Error(`[AgentProjection] Invalid state transition: agent_dismissed for missing indexed agent ${resolvedAgentId}`)
      }

      if (existing.status === 'dismissed') {
        return state
      }

      emit.agentDismissed({
        forkId: existing.forkId,
        parentForkId: existing.parentForkId,
        agentId: existing.agentId,
        name: existing.name,
        role: existing.role,
        taskId: existing.taskId,
        result: event.result,
        reason: event.reason,
      })

      return {
        ...state,
        agents: new Map(state.agents).set(resolvedAgentId, {
          ...existing,
          status: 'dismissed',
          result: event.result,
          dismissReason: event.reason,
          dismissedAt: event.timestamp,
        }),
      }
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

      let nextState: AgentState = { ...state, pendingMessages }

      if (entry.dest === 'parent' && entry.forkId !== null) {
        const existing = state.deferredParentMessages.get(entry.forkId) ?? []
        const deferredParentMessages = new Map(state.deferredParentMessages)
        deferredParentMessages.set(entry.forkId, [...existing, { text: entry.text }])
        nextState = { ...nextState, deferredParentMessages }
      }

      if (entry.dest !== 'user' && entry.dest !== 'parent') {
        const target = getActiveAgent(state, entry.dest)
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

      const agent = getAgentByForkId(state, event.forkId)
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