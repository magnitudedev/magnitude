/**
 * AgentStatusProjection
 *
 * Tracks agent identity, metadata, and execution lifecycle status.
 */

import { Projection, Signal } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { WorkingStateProjection } from './working-state'

export type AgentStatus = 'starting' | 'working' | 'idle' | 'killed'

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

export interface AgentKilledSignal {
  readonly agentId: string
  readonly forkId: string
  readonly parentForkId: string | null
  readonly type: string
  readonly title: string
  readonly reason: string
  readonly timestamp: number
}

export interface SubagentUserKilledSignal {
  readonly agentId: string
  readonly forkId: string
  readonly parentForkId: string | null
  readonly type: string
  readonly title: string
  readonly source: 'tab_close_confirm'
  readonly timestamp: number
}

function removeKilledAgent(args: {
  forkId: string
  agentId: string
  timestamp: number
  state: AgentStatusState
}): { state: AgentStatusState; agent: AgentInfo | null } {
  const { forkId, agentId, timestamp, state } = args
  const agent = getAgentByForkId(state, forkId)
  if (!agent) return { state, agent: null }
  if (agent.agentId !== agentId) return { state, agent: null }

  const nextAgents = new Map(state.agents)
  nextAgents.delete(agent.agentId)
  const nextByFork = new Map(state.agentByForkId)
  nextByFork.delete(forkId)

  return {
    state: {
      ...state,
      agents: nextAgents,
      agentByForkId: nextByFork,
    },
    agent,
  }
}

export function getAgentByForkId(state: AgentStatusState, forkId: string): AgentInfo | undefined {
  const agentId = state.agentByForkId.get(forkId)
  if (!agentId) return undefined
  return state.agents.get(agentId)
}

export function getActiveAgent(state: AgentStatusState, agentId: string): AgentInfo | undefined {
  return state.agents.get(agentId)
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
    agentBecameIdle: Signal.create<AgentBecameIdleSignal>('AgentStatus/agentBecameIdle'),
    agentBecameWorking: Signal.create<AgentBecameWorkingSignal>('AgentStatus/agentBecameWorking'),
    agentKilled: Signal.create<AgentKilledSignal>('AgentStatus/agentKilled'),
    subagentUserKilled: Signal.create<SubagentUserKilledSignal>('AgentStatus/subagentUserKilled'),
  },

  eventHandlers: {
    agent_created: ({ event, state, emit }) => {
      const normalizedMode: 'clone' | 'spawn' = event.mode === 'clone' ? 'clone' : 'spawn'
      const normalizedContext = typeof event.context === 'string' ? event.context : ''
      const normalizedTaskId = typeof event.taskId === 'string' && event.taskId.trim().length > 0
        ? event.taskId
        : `legacy-${event.agentId}-${event.forkId}`

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
        taskId: normalizedTaskId,
        mode: normalizedMode,
        context: normalizedContext,
        timestamp: event.timestamp,
      })

      const agent: AgentInfo = {
        agentId: event.agentId,
        forkId: event.forkId,
        parentForkId: event.parentForkId,
        name: event.name,
        role: event.role,
        context: normalizedContext,
        mode: normalizedMode,
        taskId: normalizedTaskId,
        message: event.message ?? null,
        status: 'starting',
      }

      return {
        ...state,
        agents: new Map(state.agents).set(event.agentId, agent),
        agentByForkId: new Map(state.agentByForkId).set(event.forkId, event.agentId),
      }
    },


    turn_started: ({ event, state, emit }) => {
      if (event.forkId === null) return state

      const agent = getAgentByForkId(state, event.forkId)
      if (!agent) return state

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
      if (!agent) return state

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
      if (!agent) return state

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

    agent_killed: ({ event, state, emit }) => {
      const removed = removeKilledAgent({
        forkId: event.forkId,
        agentId: event.agentId,
        timestamp: event.timestamp,
        state,
      })
      if (!removed.agent) return state

      emit.agentKilled({
        agentId: removed.agent.agentId,
        forkId: removed.agent.forkId,
        parentForkId: removed.agent.parentForkId,
        type: removed.agent.role,
        title: removed.agent.name,
        reason: event.reason,
        timestamp: event.timestamp,
      })

      return removed.state
    },

    subagent_user_killed: ({ event, state, emit }) => {
      const removed = removeKilledAgent({
        forkId: event.forkId,
        agentId: event.agentId,
        timestamp: event.timestamp,
        state,
      })
      if (!removed.agent) return state

      emit.subagentUserKilled({
        agentId: removed.agent.agentId,
        forkId: removed.agent.forkId,
        parentForkId: removed.agent.parentForkId,
        type: removed.agent.role,
        title: removed.agent.name,
        source: event.source,
        timestamp: event.timestamp,
      })

      return removed.state
    },
  },

  signalHandlers: (on) => [
    on(WorkingStateProjection.signals.forkBecameStable, ({ value, state, emit }) => {
      if (value.forkId === null) return state

      const agent = getAgentByForkId(state, value.forkId)
      if (!agent) return state

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