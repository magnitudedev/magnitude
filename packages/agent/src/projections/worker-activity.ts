import { Projection, Signal } from '@magnitudedev/event-core'
import { Schema } from 'effect'
import type { AppEvent } from '../events'
import { AgentLifecycleProjection, getAgentByForkId } from './agent-lifecycle'

const TurnEntrySchema = Schema.Struct({
  forkId: Schema.String,
  agentId: Schema.String,
  turnId: Schema.String,
  prose: Schema.NullOr(Schema.String),
})
export type TurnEntry = typeof TurnEntrySchema.Type

export const SubagentActivityStateSchema = Schema.Struct({
  entriesByParent: Schema.ReadonlyMap({ key: Schema.NullOr(Schema.String), value: Schema.Array(TurnEntrySchema) }),
  seenCursorByParent: Schema.ReadonlyMap({ key: Schema.NullOr(Schema.String), value: Schema.Number }),
  pendingProse: Schema.ReadonlyMap({ key: Schema.String, value: Schema.String }),
  userMessageIdsByFork: Schema.ReadonlyMap({ key: Schema.String, value: Schema.ReadonlySet(Schema.String) }),
})
export type SubagentActivityState = typeof SubagentActivityStateSchema.Type

export const WorkerActivityProjection = Projection.define<AppEvent>()({
  name: 'SubagentActivity',
  state: SubagentActivityStateSchema,

  reads: [AgentLifecycleProjection] as const,

  initial: {
    entriesByParent: new Map<string | null, readonly TurnEntry[]>(),
    seenCursorByParent: new Map<string | null, number>(),
    pendingProse: new Map<string, string>(),
    userMessageIdsByFork: new Map<string, ReadonlySet<string>>(),

  },

  signals: {
    unseenActivityAvailable: Signal.create<{ parentForkId: string | null; entries: readonly TurnEntry[] }>('SubagentActivity/unseenActivityAvailable'),
  },

  eventHandlers: {
    message_start: ({ event, state }) => {
      if (event.forkId === null) return state
      if (event.destination.kind !== 'coordinator') return state

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

    turn_outcome: ({ event, state, read }) => {
      if (event.forkId === null) return state

      const agentState = read(AgentLifecycleProjection)
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

    worker_user_killed: ({ event, state }) => {
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

    worker_idle_closed: ({ event, state }) => {
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
