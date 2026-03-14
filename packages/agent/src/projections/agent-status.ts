/**
 * AgentStatusProjection
 *
 * Tracks execution lifecycle status for agents.
 */

import { Projection, Signal } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { AgentRoutingProjection, getAgentByForkId } from './agent-routing'
import { WorkingStateProjection } from './working-state'

export type AgentStatus = 'starting' | 'working' | 'idle' | 'dismissed'

export interface AgentStatusState {
  readonly statuses: ReadonlyMap<string, AgentStatus>
}

export interface AgentBecameIdleSignal {
  readonly agentId: string
  readonly forkId: string
  readonly type: string
  readonly parentForkId: string | null
  readonly reason: 'stable' | 'interrupt' | 'error'
  readonly timestamp: number
}

export interface AgentBecameWorkingSignal {
  readonly agentId: string
  readonly forkId: string
  readonly type: string
  readonly parentForkId: string | null
  readonly timestamp: number
}

function getStatus(state: AgentStatusState, agentId: string): AgentStatus | undefined {
  return state.statuses.get(agentId)
}

function isDismissed(state: AgentStatusState, agentId: string): boolean {
  return getStatus(state, agentId) === 'dismissed'
}

export const AgentStatusProjection = Projection.define<AppEvent, AgentStatusState>()(({
  name: 'AgentStatus',
  reads: [AgentRoutingProjection, WorkingStateProjection] as const,

  initial: {
    statuses: new Map(),
  },

  signals: {
    agentBecameIdle: Signal.create<AgentBecameIdleSignal>('AgentStatus/agentBecameIdle'),
    agentBecameWorking: Signal.create<AgentBecameWorkingSignal>('AgentStatus/agentBecameWorking'),
  },

  eventHandlers: {
    agent_created: ({ event, state }) => ({
      ...state,
      statuses: new Map(state.statuses).set(event.agentId, 'starting'),
    }),

    agent_dismissed: ({ event, state }) => {
      const resolvedAgentId = state.statuses.has(event.agentId) ? event.agentId : undefined
      if (!resolvedAgentId || isDismissed(state, resolvedAgentId)) return state
      return {
        ...state,
        statuses: new Map(state.statuses).set(resolvedAgentId, 'dismissed'),
      }
    },

    agent_paused: ({ event, state }) => {
      if (isDismissed(state, event.agentId)) return state
      return {
        ...state,
        statuses: new Map(state.statuses).set(event.agentId, 'idle'),
      }
    },

    turn_started: ({ event, state, emit, read }) => {
      if (event.forkId === null) return state

      const routing = read(AgentRoutingProjection)
      const agent = getAgentByForkId(routing, event.forkId)
      if (!agent || isDismissed(state, agent.agentId)) return state

      if (getStatus(state, agent.agentId) !== 'working') {
        emit.agentBecameWorking({
          agentId: agent.agentId,
          forkId: agent.forkId,
          type: agent.role,
          parentForkId: agent.parentForkId,
          timestamp: event.timestamp,
        })
      }

      return {
        ...state,
        statuses: new Map(state.statuses).set(agent.agentId, 'working'),
      }
    },

    turn_unexpected_error: ({ event, state, emit, read }) => {
      if (event.forkId === null) return state

      const routing = read(AgentRoutingProjection)
      const agent = getAgentByForkId(routing, event.forkId)
      if (!agent || isDismissed(state, agent.agentId)) return state

      if (getStatus(state, agent.agentId) !== 'idle') {
        emit.agentBecameIdle({
          agentId: agent.agentId,
          forkId: agent.forkId,
          type: agent.role,
          parentForkId: agent.parentForkId,
          reason: 'error',
          timestamp: event.timestamp,
        })
      }

      return {
        ...state,
        statuses: new Map(state.statuses).set(agent.agentId, 'idle'),
      }
    },

    interrupt: ({ event, state, emit, read }) => {
      if (event.forkId === null) return state

      const routing = read(AgentRoutingProjection)
      const agent = getAgentByForkId(routing, event.forkId)
      if (!agent || isDismissed(state, agent.agentId)) return state

      if (getStatus(state, agent.agentId) !== 'idle') {
        emit.agentBecameIdle({
          agentId: agent.agentId,
          forkId: agent.forkId,
          type: agent.role,
          parentForkId: agent.parentForkId,
          reason: 'interrupt',
          timestamp: event.timestamp,
        })
      }

      return {
        ...state,
        statuses: new Map(state.statuses).set(agent.agentId, 'idle'),
      }
    },
  },

  signalHandlers: (on) => [






    on(WorkingStateProjection.signals.forkBecameStable, ({ value, state, emit, read }) => {
      if (value.forkId === null) return state

      const routing = read(AgentRoutingProjection)
      const agent = getAgentByForkId(routing, value.forkId)
      if (!agent || isDismissed(state, agent.agentId)) return state

      if (getStatus(state, agent.agentId) !== 'idle') {
        emit.agentBecameIdle({
          agentId: agent.agentId,
          forkId: agent.forkId,
          type: agent.role,
          parentForkId: agent.parentForkId,
          reason: 'stable',
          timestamp: value.timestamp,
        })
      }

      return {
        ...state,
        statuses: new Map(state.statuses).set(agent.agentId, 'idle'),
      }
    }),
  ],
}))