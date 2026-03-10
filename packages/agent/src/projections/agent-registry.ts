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
import { ForkProjection } from './fork'

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

interface AgentStatusChangedBase {
  readonly agentId: string
  readonly agentType: string
  readonly previousStatus: AgentStatus
  readonly parentForkId: string | null
}

export type AgentStatusChangedSignal =
  | AgentStatusChangedBase & { readonly status: 'idle'; readonly reason: 'stable' | 'interrupt' }
  | AgentStatusChangedBase & { readonly status: 'running' }
  | AgentStatusChangedBase & { readonly status: 'paused' }
  | AgentStatusChangedBase & { readonly status: 'dismissed' }

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
    agent_created: ({ event, state, emit, read }) => {
      const agents = new Map(state.agents)
      agents.set(event.agentId, {
        agentId: event.agentId,
        type: event.agentType,
        taskId: event.taskId,
        forkId: event.agentForkId,
        status: 'running',
        message: event.message ?? null,
      })
      const parentForkId = read(ForkProjection).forks.get(event.agentForkId)?.parentForkId ?? null
      emit.agentStatusChanged({ agentId: event.agentId, agentType: event.agentType, status: 'running', previousStatus: 'running', parentForkId })
      return { ...state, agents }
    },

    agent_paused: ({ event, state, emit, read }) => {
      const existing = state.agents.get(event.agentId)
      if (!existing) return state

      const agents = new Map(state.agents)
      agents.set(event.agentId, { ...existing, status: 'paused' })
      const parentForkId = read(ForkProjection).forks.get(existing.forkId)?.parentForkId ?? null
      emit.agentStatusChanged({ agentId: event.agentId, agentType: existing.type, status: 'paused', previousStatus: existing.status, parentForkId })
      return { ...state, agents }
    },

    agent_dismissed: ({ event, state, emit, read }) => {
      const existing = state.agents.get(event.agentId)
      if (!existing) return state

      const agents = new Map(state.agents)
      agents.set(event.agentId, { ...existing, status: 'dismissed' })
      const parentForkId = read(ForkProjection).forks.get(existing.forkId)?.parentForkId ?? null
      emit.agentStatusChanged({ agentId: event.agentId, agentType: existing.type, status: 'dismissed', previousStatus: existing.status, parentForkId })
      return { ...state, agents }
    },

    interrupt: ({ event, state, emit, read }) => {
      if (event.forkId === null) return state

      for (const [agentId, entry] of state.agents) {
        if (entry.forkId === event.forkId && entry.status === 'running') {
          const parentForkId = read(ForkProjection).forks.get(entry.forkId)?.parentForkId ?? null
          const agents = new Map(state.agents)
          agents.set(agentId, { ...entry, status: 'idle' })
          emit.agentStatusChanged({ agentId, agentType: entry.type, status: 'idle', previousStatus: entry.status, parentForkId, reason: 'interrupt' })
          return { ...state, agents }
        }
      }

      return state
    },

    turn_started: ({ event, state, emit, read }) => {
      if (event.forkId === null) return state

      for (const [agentId, entry] of state.agents) {
        if (entry.forkId === event.forkId && (entry.status === 'idle' || entry.status === 'paused')) {
          const agents = new Map(state.agents)
          agents.set(agentId, { ...entry, status: 'running' })
          const parentForkId = read(ForkProjection).forks.get(entry.forkId)?.parentForkId ?? null
          emit.agentStatusChanged({ agentId, agentType: entry.type, status: 'running', previousStatus: entry.status, parentForkId })
          return { ...state, agents }
        }
      }

      return state
    },
  },

  reads: [WorkingStateProjection, ForkProjection] as const,

  signalHandlers: (on) => [
    on(WorkingStateProjection.signals.forkBecameStable, ({ value, state, emit, read }) => {
      const { forkId } = value
      for (const [agentId, entry] of state.agents) {
        if (entry.forkId === forkId && entry.status === 'running') {
          const parentForkId = read(ForkProjection).forks.get(entry.forkId)?.parentForkId ?? null
          const agents = new Map(state.agents)
          agents.set(agentId, { ...entry, status: 'idle' })
          emit.agentStatusChanged({ agentId, agentType: entry.type, status: 'idle', previousStatus: entry.status, parentForkId, reason: 'stable' })
          return { ...state, agents }
        }
      }
      return state
    }),
  ],
})
