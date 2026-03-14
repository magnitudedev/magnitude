import { Projection, Signal } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { AgentStatusProjection, getAgentByForkId } from './agent-status'
import { WorkingStateProjection } from './working-state'

export interface TurnEntry {
  readonly forkId: string
  readonly agentId: string
  readonly turnId: string
  readonly prose: string | null
  readonly toolsCalled: readonly string[]
  readonly artifactsWritten: readonly string[]
}

export interface SubagentActivityState {
  readonly entriesByParent: ReadonlyMap<string | null, readonly TurnEntry[]>
  readonly seenCursorByParent: ReadonlyMap<string | null, number>
  // Buffer prose chunks during a turn (message_chunk events for user-dest messages)
  readonly pendingProse: ReadonlyMap<string, string>
  // Track user-destination message ids by fork
  readonly userMessageIdsByFork: ReadonlyMap<string, ReadonlySet<string>>
  // Buffer artifact events that arrive during a turn
  readonly pendingArtifacts: ReadonlyMap<string, { written: readonly string[] }>
}

export const SubagentActivityProjection = Projection.define<AppEvent, SubagentActivityState>()({
  name: 'SubagentActivity',

  reads: [AgentStatusProjection] as const,

  initial: {
    entriesByParent: new Map(),
    seenCursorByParent: new Map(),
    pendingProse: new Map(),
    userMessageIdsByFork: new Map(),
    pendingArtifacts: new Map(),
  },

  signals: {
    unseenActivityAvailable: Signal.create<{ parentForkId: string | null; entries: readonly TurnEntry[] }>('SubagentActivity/unseenActivityAvailable'),
  },

  eventHandlers: {
    message_start: ({ event, state }) => {
      if (event.forkId === null) return state
      if (event.dest !== 'user') return state

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

    artifact_changed: ({ event, state }) => {
      if (event.forkId === null) return state

      const forkId = event.forkId
      const existing = state.pendingArtifacts.get(forkId) ?? { written: [] }
      return {
        ...state,
        pendingArtifacts: new Map(state.pendingArtifacts).set(forkId, {
          ...existing,
          written: [...existing.written, event.id],
        }),
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
      const toolsCalled = event.toolCalls.map(tc => tc.toolName)

      // Collect pending artifact activity for this fork
      const pendingArtifact = state.pendingArtifacts.get(event.forkId) ?? { written: [] }

      const entry: TurnEntry = {
        forkId: event.forkId,
        agentId: agent.name,
        turnId: event.turnId,
        prose,
        toolsCalled,
        artifactsWritten: pendingArtifact.written,
      }

      const parentForkId = agent.parentForkId
      const existing = state.entriesByParent.get(parentForkId) ?? []

      // Clear pending state for this fork
      const newPendingProse = new Map(state.pendingProse)
      newPendingProse.delete(event.forkId)
      const newPendingArtifacts = new Map(state.pendingArtifacts)
      newPendingArtifacts.delete(event.forkId)

      const newUserMessageIdsByFork = new Map(state.userMessageIdsByFork)
      newUserMessageIdsByFork.delete(event.forkId)

      return {
        ...state,
        entriesByParent: new Map(state.entriesByParent).set(parentForkId, [...existing, entry]),
        pendingProse: newPendingProse,
        userMessageIdsByFork: newUserMessageIdsByFork,
        pendingArtifacts: newPendingArtifacts,
      }
    },
  },

  signalHandlers: (on) => [
    on(WorkingStateProjection.signals.shouldTriggerChanged, ({ value, state, emit }) => {
      if (!value.shouldTrigger) return state

      const parentForkId = value.forkId
      const entries = state.entriesByParent.get(parentForkId) ?? []
      const cursor = state.seenCursorByParent.get(parentForkId) ?? 0

      if (entries.length <= cursor) return state

      const unseen = entries.slice(cursor)
      emit.unseenActivityAvailable({ parentForkId, entries: unseen })

      return {
        ...state,
        seenCursorByParent: new Map(state.seenCursorByParent).set(parentForkId, entries.length),
      }
    }),
  ],
})