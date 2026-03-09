/**
 * AgentRegistryProjection
 *
 * Global projection tracking active agents.
 * Replaces the task-graph's activeAgent tracking.
 * Tracks agent ID, type, goal, status, and fork ID.
 */

import { Projection, Signal } from '@magnitudedev/event-core'
import type { AppEvent, WorkAgentType } from '../events'
import { WorkingStateProjection } from './working-state'

// =============================================================================
// Types
// =============================================================================

export type AgentStatus = 'running' | 'idle' | 'paused' | 'dismissed'

export interface AgentEntry {
  readonly agentId: string
  readonly type: WorkAgentType
  readonly taskId: string
  readonly forkId: string
  readonly status: AgentStatus
  readonly message: string | null
}

export interface AgentRegistryState {
  readonly agents: ReadonlyMap<string, AgentEntry>
}

// =============================================================================
// Signal Types
// =============================================================================

export interface AgentStatusChangedSignal {
  readonly agentId: string
  readonly status: AgentStatus
  readonly previousStatus: AgentStatus
}

export const agentStatusChangedSignal = Signal.create<AgentStatusChangedSignal>('AgentRegistry/agentStatusChanged')

// =============================================================================
// Projection
// =============================================================================

export const AgentRegistryProjection = Projection.define<AppEvent, AgentRegistryState>()({
  name: 'AgentRegistry',

  initial: {
    agents: new Map(),
  },

  signals: {
    agentStatusChanged: agentStatusChangedSignal,
  },

  eventHandlers: {
    agent_created: ({ event, state, emit }) => {
      const agents = new Map(state.agents)
      agents.set(event.agentId, {
        agentId: event.agentId,
        type: event.agentType,
        taskId: event.taskId,
        forkId: event.agentForkId,
        status: 'running',
        message: event.message ?? null,
      })
      emit.agentStatusChanged({ agentId: event.agentId, status: 'running', previousStatus: 'running' })
      return { ...state, agents }
    },

    agent_paused: ({ event, state, emit }) => {
      const existing = state.agents.get(event.agentId)
      if (!existing) return state

      const agents = new Map(state.agents)
      agents.set(event.agentId, { ...existing, status: 'paused' })
      emit.agentStatusChanged({ agentId: event.agentId, status: 'paused', previousStatus: existing.status })
      return { ...state, agents }
    },

    agent_dismissed: ({ event, state, emit }) => {
      const existing = state.agents.get(event.agentId)
      if (!existing) return state

      const agents = new Map(state.agents)
      agents.set(event.agentId, { ...existing, status: 'dismissed' })
      emit.agentStatusChanged({ agentId: event.agentId, status: 'dismissed', previousStatus: existing.status })
      return { ...state, agents }
    },

    interrupt: ({ event, state, emit }) => {
      if (event.forkId === null) return state

      for (const [agentId, entry] of state.agents) {
        if (entry.forkId === event.forkId && entry.status === 'running') {
          const agents = new Map(state.agents)
          agents.set(agentId, { ...entry, status: 'idle' })
          emit.agentStatusChanged({ agentId, status: 'idle', previousStatus: entry.status })
          return { ...state, agents }
        }
      }

      return state
    },

    turn_started: ({ event, state, emit }) => {
      if (event.forkId === null) return state

      for (const [agentId, entry] of state.agents) {
        if (entry.forkId === event.forkId && (entry.status === 'idle' || entry.status === 'paused')) {
          const agents = new Map(state.agents)
          agents.set(agentId, { ...entry, status: 'running' })
          emit.agentStatusChanged({ agentId, status: 'running', previousStatus: entry.status })
          return { ...state, agents }
        }
      }

      return state
    },
  },

  reads: [WorkingStateProjection] as const,

  signalHandlers: (on) => [
    on(WorkingStateProjection.signals.forkBecameStable, ({ value, state, emit }) => {
      const { forkId } = value
      for (const [agentId, entry] of state.agents) {
        if (entry.forkId === forkId && entry.status === 'running') {
          const agents = new Map(state.agents)
          agents.set(agentId, { ...entry, status: 'idle' })
          emit.agentStatusChanged({ agentId, status: 'idle', previousStatus: entry.status })
          return { ...state, agents }
        }
      }
      return state
    }),
  ],
})
