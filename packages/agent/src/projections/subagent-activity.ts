import { Projection, Signal } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { AgentStatusProjection, getAgentByForkId } from './agent-status'



export interface TurnEntry {
  readonly forkId: string
  readonly agentId: string
  readonly turnId: string
  readonly prose: string | null
}

export interface SubagentActivityState {
  readonly entriesByParent: ReadonlyMap<string | null, readonly TurnEntry[]>
  readonly seenCursorByParent: ReadonlyMap<string | null, number>
  // Buffer prose chunks during a turn (message_chunk events for top-level messages)
  readonly pendingProse: ReadonlyMap<string, string>
  // Track top-level message ids by fork
  readonly userMessageIdsByFork: ReadonlyMap<string, ReadonlySet<string>>

}

export const SubagentActivityProjection = Projection.define<AppEvent, SubagentActivityState>()({
  name: 'SubagentActivity',

  reads: [AgentStatusProjection] as const,

  initial: {
    entriesByParent: new Map(),
    seenCursorByParent: new Map(),
    pendingProse: new Map(),
    userMessageIdsByFork: new Map(),

  },

  signals: {
    unseenActivityAvailable: Signal.create<{ parentForkId: string | null; entries: readonly TurnEntry[] }>('SubagentActivity/unseenActivityAvailable'),
  },

  eventHandlers: {
    message_start: ({ event, state }) => {
      if (event.forkId === null) return state
      if (event.taskId !== null) return state

      const existing = state.userMessageIdsByFork.get(event.forkId) ?? new Set<string>()
      const next = new Set(existing)
      next.add(event.id)
      return {
        ...state,
        userMessageIdsByFork: new Map(state.userMessageIdsByFork).set(event.forkId, next),
      }
    },

    message_chunk: ({ event, state }) => {
      if (event.forkId === null) return state
      const ids = state.userMessageIdsByFork.get(event.forkId)
      if (!ids?.has(event.id)) return state

      const existing = state.pendingProse.get(event.forkId) ?? ''
      return {
        ...state,
        pendingProse: new Map(state.pendingProse).set(event.forkId, existing + event.text),
      }
    },

    turn_started: ({ event, state, emit }) => {
      const parentForkId = event.forkId
      const entries = state.entriesByParent.get(parentForkId) ?? []
      const cursor = state.seenCursorByParent.get(parentForkId) ?? 0

      if (entries.length <= cursor) return state

      const unseen = entries.slice(cursor)
      emit.unseenActivityAvailable({ parentForkId, entries: unseen })

      return {
        ...state,
        seenCursorByParent: new Map(state.seenCursorByParent).set(parentForkId, entries.length),
      }
    },

    turn_completed: ({ event, state, read }) => {
      if (event.forkId === null) return state

      const agentState = read(AgentStatusProjection)
      const agent = getAgentByForkId(agentState, event.forkId)
      if (!agent) return state

      // Get accumulated prose from message_chunk events
      const rawProse = state.pendingProse.get(event.forkId) ?? ''
      const prose = rawProse.trim() || null
      const entry: TurnEntry = {
        forkId: event.forkId,
        agentId: agent.agentId,
        turnId: event.turnId,
        prose,
      }

      const parentForkId = agent.parentForkId
      const existing = state.entriesByParent.get(parentForkId) ?? []

      // Clear pending state for this fork
      const newPendingProse = new Map(state.pendingProse)
      newPendingProse.delete(event.forkId)
      const newUserMessageIdsByFork = new Map(state.userMessageIdsByFork)
      newUserMessageIdsByFork.delete(event.forkId)

      return {
        ...state,
        entriesByParent: new Map(state.entriesByParent).set(parentForkId, [...existing, entry]),
        pendingProse: newPendingProse,
        userMessageIdsByFork: newUserMessageIdsByFork,
      }
    },

    turn_unexpected_error: ({ event, state }) => {
      if (event.forkId === null) return state

      const pendingProse = new Map(state.pendingProse)
      pendingProse.delete(event.forkId)

      const userMessageIdsByFork = new Map(state.userMessageIdsByFork)
      userMessageIdsByFork.delete(event.forkId)

      return {
        ...state,
        pendingProse,
        userMessageIdsByFork,
      }
    },

    agent_killed: ({ event, state }) => {
      const pendingProse = new Map(state.pendingProse)
      pendingProse.delete(event.forkId)

      const userMessageIdsByFork = new Map(state.userMessageIdsByFork)
      userMessageIdsByFork.delete(event.forkId)

      return {
        ...state,
        pendingProse,
        userMessageIdsByFork,
      }
    },

    subagent_user_killed: ({ event, state }) => {
      const pendingProse = new Map(state.pendingProse)
      pendingProse.delete(event.forkId)

      const userMessageIdsByFork = new Map(state.userMessageIdsByFork)
      userMessageIdsByFork.delete(event.forkId)

      return {
        ...state,
        pendingProse,
        userMessageIdsByFork,
      }
    },

    subagent_idle_closed: ({ event, state }) => {
      const pendingProse = new Map(state.pendingProse)
      pendingProse.delete(event.forkId)

      const userMessageIdsByFork = new Map(state.userMessageIdsByFork)
      userMessageIdsByFork.delete(event.forkId)

      return {
        ...state,
        pendingProse,
        userMessageIdsByFork,
      }
    },
  },

})