import { Projection, Signal } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { AgentProjection, getAgentByForkId } from './agent'
import { WorkingStateProjection } from './working-state'

export interface AgentBecameIdleSignal {
  readonly agentId: string
  readonly forkId: string
  readonly type: string
  readonly parentForkId: string | null
  readonly reason: 'stable' | 'interrupt'
}

export interface AgentResumedSignal {
  readonly agentId: string
  readonly forkId: string
  readonly type: string
  readonly parentForkId: string | null
}

export const AgentStatusBridgeProjection = Projection.define<AppEvent, {}>()({
  name: 'AgentStatus',

  reads: [AgentProjection, WorkingStateProjection] as const,

  initial: {},

  signals: {
    agentBecameIdle: Signal.create<AgentBecameIdleSignal>('AgentStatus/agentBecameIdle'),
    agentResumed: Signal.create<AgentResumedSignal>('AgentStatus/agentResumed'),
  },

  eventHandlers: {
    turn_started: ({ event, state, emit, read }) => {
      if (event.forkId === null) return state

      const agent = getAgentByForkId(read(AgentProjection), event.forkId)
      if (!agent || agent.status === 'dismissed') return state

      emit.agentResumed({
        agentId: agent.agentId,
        forkId: agent.forkId,
        type: agent.role,
        parentForkId: agent.parentForkId,
      })

      return state
    },

    interrupt: ({ event, state, emit, read }) => {
      if (event.forkId === null) return state

      const agent = getAgentByForkId(read(AgentProjection), event.forkId)
      if (!agent || agent.status === 'dismissed') return state

      emit.agentBecameIdle({
        agentId: agent.agentId,
        forkId: agent.forkId,
        type: agent.role,
        parentForkId: agent.parentForkId,
        reason: 'interrupt',
      })

      return state
    },
  },

  signalHandlers: (on) => [
    on(WorkingStateProjection.signals.forkBecameStable, ({ value, state, emit, read }) => {
      if (value.forkId === null) return state

      const agent = getAgentByForkId(read(AgentProjection), value.forkId)
      if (!agent || agent.status === 'dismissed') return state

      emit.agentBecameIdle({
        agentId: agent.agentId,
        forkId: agent.forkId,
        type: agent.role,
        parentForkId: agent.parentForkId,
        reason: 'stable',
      })

      return state
    }),
  ],
})