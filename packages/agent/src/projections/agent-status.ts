/**
 * AgentStatusProjection
 *
 * Tracks agent identity, metadata, and execution lifecycle status.
 */

import { Projection, Signal } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { WorkingStateProjection } from './working-state'

export type AgentStatus = 'starting' | 'working' | 'idle' | 'dismissed'

export interface AgentInfo {
  readonly agentId: string
  readonly forkId: string
  readonly parentForkId: string | null
  readonly name: string
  readonly role: string
  readonly context: string
  readonly mode: 'clone' | 'spawn'
  readonly taskId: string
  readonly message: string | null
  readonly status: AgentStatus
  readonly result?: unknown
  readonly dismissReason?: 'dismissed' | 'interrupted' | 'completed'
}

export interface AgentStatusState {
  readonly agents: ReadonlyMap<string, AgentInfo>
  readonly agentByForkId: ReadonlyMap<string, string>
}

export interface AgentCreatedSignal {
  readonly forkId: string
  readonly parentForkId: string | null
  readonly agentId: string
  readonly name: string
  readonly role: string
  readonly type: string
  readonly taskId: string
  readonly mode: 'clone' | 'spawn'
  readonly context: string
  readonly timestamp: number
}

export interface AgentDismissedSignal {
  readonly forkId: string
  readonly parentForkId: string | null
  readonly agentId: string
  readonly name: string
  readonly role: string
  readonly type: string
  readonly taskId: string
  readonly result: unknown
  readonly reason: 'dismissed' | 'interrupted' | 'completed'
  readonly timestamp: number
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
  return state.agents.get(agentId)?.status
}

export function getAgentByForkId(state: AgentStatusState, forkId: string): AgentInfo | undefined {
  const agentId = state.agentByForkId.get(forkId)
  if (!agentId) return undefined
  return state.agents.get(agentId)
}

export function getActiveAgent(state: AgentStatusState, agentId: string): AgentInfo | undefined {
  const agent = state.agents.get(agentId)
  if (!agent || agent.status === 'dismissed') return undefined
  return agent
}

function isDismissed(state: AgentStatusState, agentId: string): boolean {
  return getStatus(state, agentId) === 'dismissed'
}

export const AgentStatusProjection = Projection.define<AppEvent, AgentStatusState>()(({
  name: 'AgentStatus',
  reads: [WorkingStateProjection] as const,

  initial: {
    agents: new Map(),
    agentByForkId: new Map(),
  },

  signals: {
    agentCreated: Signal.create<AgentCreatedSignal>('AgentStatus/created'),
    agentDismissed: Signal.create<AgentDismissedSignal>('AgentStatus/dismissed'),
    agentBecameIdle: Signal.create<AgentBecameIdleSignal>('AgentStatus/agentBecameIdle'),
    agentBecameWorking: Signal.create<AgentBecameWorkingSignal>('AgentStatus/agentBecameWorking'),
  },

  eventHandlers: {
    agent_created: ({ event, state, emit }) => {
      const existingAgent = state.agents.get(event.agentId)
      if (existingAgent) {
        throw new Error(`[AgentStatusProjection] Invalid state transition: agent_created for already existing agent ${event.agentId} (forkId: ${existingAgent.forkId})`)
      }

      const existingForkAgentId = state.agentByForkId.get(event.forkId)
      if (existingForkAgentId) {
        throw new Error(`[AgentStatusProjection] Invalid state transition: agent_created for already indexed fork ${event.forkId} (agentId: ${existingForkAgentId})`)
      }

      emit.agentCreated({
        forkId: event.forkId,
        parentForkId: event.parentForkId,
        agentId: event.agentId,
        name: event.name,
        role: event.role,
        type: event.role,
        taskId: event.taskId,
        mode: event.mode,
        context: event.context,
        timestamp: event.timestamp,
      })

      const agent: AgentInfo = {
        agentId: event.agentId,
        forkId: event.forkId,
        parentForkId: event.parentForkId,
        name: event.name,
        role: event.role,
        context: event.context,
        mode: event.mode,
        taskId: event.taskId,
        message: event.message ?? null,
        status: 'starting',
      }

      return {
        ...state,
        agents: new Map(state.agents).set(event.agentId, agent),
        agentByForkId: new Map(state.agentByForkId).set(event.forkId, event.agentId),
      }
    },

    agent_dismissed: ({ event, state, emit }) => {
      const resolvedAgentId = state.agents.has(event.agentId)
        ? event.agentId
        : state.agentByForkId.get(event.forkId)

      if (!resolvedAgentId) {
        throw new Error(`[AgentStatusProjection] Invalid state transition: agent_dismissed for unknown agent ${event.agentId} (forkId: ${event.forkId})`)
      }

      const existing = state.agents.get(resolvedAgentId)
      if (!existing) {
        throw new Error(`[AgentStatusProjection] Invalid state transition: agent_dismissed for missing indexed agent ${resolvedAgentId}`)
      }

      if (existing.status === 'dismissed') return state

      emit.agentDismissed({
        forkId: existing.forkId,
        parentForkId: existing.parentForkId,
        agentId: existing.agentId,
        name: existing.name,
        role: existing.role,
        type: existing.role,
        taskId: existing.taskId,
        result: event.result,
        reason: event.reason,
        timestamp: event.timestamp,
      })

      return {
        ...state,
        agents: new Map(state.agents).set(resolvedAgentId, {
          ...existing,
          status: 'dismissed',
          result: event.result,
          dismissReason: event.reason,
        }),
      }
    },

    turn_started: ({ event, state, emit }) => {
      if (event.forkId === null) return state

      const agent = getAgentByForkId(state, event.forkId)
      if (!agent || isDismissed(state, agent.agentId)) return state

      if (agent.status !== 'working') {
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
        agents: new Map(state.agents).set(agent.agentId, { ...agent, status: 'working' }),
      }
    },

    turn_unexpected_error: ({ event, state, emit }) => {
      if (event.forkId === null) return state

      const agent = getAgentByForkId(state, event.forkId)
      if (!agent || isDismissed(state, agent.agentId)) return state

      if (agent.status !== 'idle') {
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
        agents: new Map(state.agents).set(agent.agentId, { ...agent, status: 'idle' }),
      }
    },

    interrupt: ({ event, state, emit }) => {
      if (event.forkId === null) return state

      const agent = getAgentByForkId(state, event.forkId)
      if (!agent || isDismissed(state, agent.agentId)) return state

      if (agent.status !== 'idle') {
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
        agents: new Map(state.agents).set(agent.agentId, { ...agent, status: 'idle' }),
      }
    },
  },

  signalHandlers: (on) => [
    on(WorkingStateProjection.signals.forkBecameStable, ({ value, state, emit }) => {
      if (value.forkId === null) return state

      const agent = getAgentByForkId(state, value.forkId)
      if (!agent || isDismissed(state, agent.agentId)) return state

      if (agent.status !== 'idle') {
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
        agents: new Map(state.agents).set(agent.agentId, { ...agent, status: 'idle' }),
      }
    }),
  ],
}))