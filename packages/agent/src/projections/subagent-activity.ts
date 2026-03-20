import { Projection, Signal } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { AgentStatusProjection, getAgentByForkId } from './agent-status'
import { WorkingStateProjection } from './working-state'
import { SessionContextProjection } from './session-context'
import { extractWrittenFilePathFromToolEvent, isWorkspacePath } from '../workspace/file-tracking'

export interface TurnEntry {
  readonly forkId: string
  readonly agentId: string
  readonly turnId: string
  readonly prose: string | null
  readonly toolsCalled: readonly string[]
  readonly filesWritten: readonly string[]
}

export interface SubagentActivityState {
  readonly entriesByParent: ReadonlyMap<string | null, readonly TurnEntry[]>
  readonly seenCursorByParent: ReadonlyMap<string | null, number>
  // Buffer prose chunks during a turn (message_chunk events for user-dest messages)
  readonly pendingProse: ReadonlyMap<string, string>
  // Track user-destination message ids by fork
  readonly userMessageIdsByFork: ReadonlyMap<string, ReadonlySet<string>>
  // Buffer workspace file writes that arrive during a turn
  readonly pendingFiles: ReadonlyMap<string, { written: readonly string[] }>
}

export const SubagentActivityProjection = Projection.define<AppEvent, SubagentActivityState>()({
  name: 'SubagentActivity',

  reads: [AgentStatusProjection, SessionContextProjection] as const,

  initial: {
    entriesByParent: new Map(),
    seenCursorByParent: new Map(),
    pendingProse: new Map(),
    userMessageIdsByFork: new Map(),
    pendingFiles: new Map(),
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

    tool_event: ({ event, state, read }) => {
      if (event.forkId === null) return state
      if (event.event._tag !== 'ToolExecutionEnded') return state
      if (event.event.result._tag !== 'Success') return state

      const writtenPath = extractWrittenFilePathFromToolEvent(event)
      const workspacePath = read(SessionContextProjection).context?.workspacePath
      if (!workspacePath) return state // Session not yet initialized
      if (!writtenPath || !isWorkspacePath(writtenPath, workspacePath)) return state

      const forkId = event.forkId
      const existing = state.pendingFiles.get(forkId) ?? { written: [] }
      return {
        ...state,
        pendingFiles: new Map(state.pendingFiles).set(forkId, {
          ...existing,
          written: [...existing.written, writtenPath],
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

      // Collect pending file activity for this fork
      const pendingFile = state.pendingFiles.get(event.forkId) ?? { written: [] }

      const entry: TurnEntry = {
        forkId: event.forkId,
        agentId: agent.name,
        turnId: event.turnId,
        prose,
        toolsCalled,
        filesWritten: pendingFile.written,
      }

      const parentForkId = agent.parentForkId
      const existing = state.entriesByParent.get(parentForkId) ?? []

      // Clear pending state for this fork
      const newPendingProse = new Map(state.pendingProse)
      newPendingProse.delete(event.forkId)
      const newPendingFiles = new Map(state.pendingFiles)
      newPendingFiles.delete(event.forkId)

      const newUserMessageIdsByFork = new Map(state.userMessageIdsByFork)
      newUserMessageIdsByFork.delete(event.forkId)

      return {
        ...state,
        entriesByParent: new Map(state.entriesByParent).set(parentForkId, [...existing, entry]),
        pendingProse: newPendingProse,
        userMessageIdsByFork: newUserMessageIdsByFork,
        pendingFiles: newPendingFiles,
      }
    },

    agent_killed: ({ event, state }) => {
      const pendingProse = new Map(state.pendingProse)
      pendingProse.delete(event.forkId)

      const userMessageIdsByFork = new Map(state.userMessageIdsByFork)
      userMessageIdsByFork.delete(event.forkId)

      const pendingFiles = new Map(state.pendingFiles)
      pendingFiles.delete(event.forkId)

      return {
        ...state,
        pendingProse,
        userMessageIdsByFork,
        pendingFiles,
      }
    },

    subagent_user_killed: ({ event, state }) => {
      const pendingProse = new Map(state.pendingProse)
      pendingProse.delete(event.forkId)

      const userMessageIdsByFork = new Map(state.userMessageIdsByFork)
      userMessageIdsByFork.delete(event.forkId)

      const pendingFiles = new Map(state.pendingFiles)
      pendingFiles.delete(event.forkId)

      return {
        ...state,
        pendingProse,
        userMessageIdsByFork,
        pendingFiles,
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