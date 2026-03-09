/**
 * ForkProjection
 *
 * Tracks fork lifecycle. NOT a forked projection - this is global state tracking all forks.
 * Emits signals for fork lifecycle events that other projections can subscribe to.
 *
 * Fork lifecycle: fork_started (running) → fork_completed → fork_removed
 */

import { Projection, Signal } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'

// =============================================================================
// Types
// =============================================================================

export interface ForkInstance {
  /** Internal runtime ID for the fork's execution context */
  readonly forkId: string
  readonly parentForkId: string | null
  /** Human-readable display title (e.g. "Haiku test 2") */
  readonly name: string
  /** LLM-facing identifier used in comms routing (e.g. "haiku-researcher-2") */
  readonly agentId: string
  readonly status: 'running' | 'completed'
  readonly context: string
  readonly outputSchema?: unknown
  readonly taskId?: string
  readonly blocking?: boolean
  readonly mode: 'clone' | 'spawn'
  readonly role: string
  readonly result?: unknown
  readonly createdAt: number
  readonly completedAt?: number
}

export interface ForkState {
  readonly forks: Map<string, ForkInstance>
  readonly pendingMessages: Map<string, { forkId: string | null; dest: string; text: string }>
  readonly deferredParentMessages: Map<string, Array<{ text: string }>>
}

// =============================================================================
// Signal Types
// =============================================================================

export interface ForkCreated {
  readonly forkId: string
  readonly parentForkId: string | null
  readonly name: string
}

export interface ForkCompletedSignal {
  readonly forkId: string
  readonly parentForkId: string | null
  readonly name: string
  readonly result: unknown
  readonly role: string
  readonly taskId?: string
}

export interface ForkRemovedSignal {
  readonly forkId: string
  readonly parentForkId: string | null
  readonly name: string
}

export interface AgentCommunicationSignal {
  readonly targetForkId: string | null
  readonly agentId: string
  readonly message: string
  readonly timestamp: number
}

// =============================================================================
// Projection
// =============================================================================

export const ForkProjection = Projection.define<AppEvent, ForkState>()(({
  name: 'Fork',

  initial: {
    forks: new Map(),
    pendingMessages: new Map(),
    deferredParentMessages: new Map(),
  },

  signals: {
    forkCreated: Signal.create<ForkCreated>('Fork/created'),
    forkCompleted: Signal.create<ForkCompletedSignal>('Fork/completed'),
    forkRemoved: Signal.create<ForkRemovedSignal>('Fork/removed'),
    /** Sub-agent → orchestrator: message */
    agentResponse: Signal.create<AgentCommunicationSignal>('Fork/agentResponse'),
    /** Orchestrator → sub-agent: message injection */
    agentMessage: Signal.create<AgentCommunicationSignal & { readonly targetForkId: string }>('Fork/agentMessage'),
  },

  eventHandlers: {
    user_message: ({ state }) => state,

    fork_started: ({ event, state, emit }) => {
      const existing = state.forks.get(event.forkId)
      if (existing) {
        throw new Error(`[ForkProjection] Invalid state transition: fork_started for already existing fork ${event.forkId} (status: ${existing.status})`)
      }

      emit.forkCreated({
        forkId: event.forkId,
        parentForkId: event.parentForkId,
        name: event.name
      })

      const instance: ForkInstance = {
        forkId: event.forkId,
        parentForkId: event.parentForkId,
        name: event.name,
        agentId: event.agentId,
        status: 'running',
        context: event.context,
        outputSchema: event.outputSchema,
        blocking: event.blocking,
        mode: event.mode,
        role: event.role,
        taskId: event.taskId,
        createdAt: event.timestamp
      }

      return {
        ...state,
        forks: new Map(state.forks).set(event.forkId, instance),
      }
    },

    fork_completed: ({ event, state, emit }) => {
      const existing = state.forks.get(event.forkId)

      if (!existing) {
        throw new Error(`[ForkProjection] Invalid state transition: fork_completed for unknown fork ${event.forkId}`)
      }

      if (existing.status !== 'running') {
        throw new Error(`[ForkProjection] Invalid state transition: fork_completed for fork ${event.forkId} in state '${existing.status}' (expected 'running')`)
      }

      emit.forkCompleted({
        forkId: event.forkId,
        parentForkId: event.parentForkId,
        name: existing.name,
        result: event.result,
        role: existing.role,
        taskId: existing.taskId,
      })

      const updated: ForkInstance = {
        ...existing,
        status: 'completed',
        result: event.result,
        completedAt: event.timestamp
      }

      return {
        ...state,
        forks: new Map(state.forks).set(event.forkId, updated)
      }
    },

    message_start: ({ event, state }) => {
      const pending = new Map(state.pendingMessages)
      pending.set(event.id, { forkId: event.forkId, dest: event.dest, text: '' })
      return { ...state, pendingMessages: pending }
    },

    message_chunk: ({ event, state }) => {
      const entry = state.pendingMessages.get(event.id)
      if (!entry) return state
      const pending = new Map(state.pendingMessages)
      pending.set(event.id, { ...entry, text: entry.text + event.text })
      return { ...state, pendingMessages: pending }
    },

    message_end: ({ event, state, emit }) => {
      const entry = state.pendingMessages.get(event.id)
      if (!entry) return state

      const pending = new Map(state.pendingMessages)
      pending.delete(event.id)

      let nextState: ForkState = { ...state, pendingMessages: pending }

      if (entry.dest === 'parent' && entry.forkId !== null) {
        const key = entry.forkId
        const existing = state.deferredParentMessages.get(key) ?? []
        const deferred = new Map(state.deferredParentMessages)
        deferred.set(key, [...existing, { text: entry.text }])
        nextState = { ...nextState, deferredParentMessages: deferred }
      }

      if (entry.dest !== 'user' && entry.dest !== 'parent') {
        const target = [...state.forks.values()].find(f => f.agentId === entry.dest)
        if (target) {
          emit.agentMessage({
            targetForkId: target.forkId,
            agentId: entry.dest,
            message: entry.text,
            timestamp: event.timestamp,
          })

        }
      }

      return nextState
    },

    turn_completed: ({ event, state, emit }) => {
      const forkId = event.forkId
      if (forkId === null) return state

      const messages = state.deferredParentMessages.get(forkId)
      if (!messages || messages.length === 0) return state

      if (!event.result.success || event.result.turnDecision !== 'yield') {
        const deferred = new Map(state.deferredParentMessages)
        deferred.delete(forkId)
        return { ...state, deferredParentMessages: deferred }
      }

      const fork = state.forks.get(forkId)
      if (!fork) return state

      const fullText = messages.map(m => m.text).join('\n').trim()

      emit.agentResponse({
        targetForkId: fork.parentForkId,
        agentId: fork.agentId,
        message: fullText,
        timestamp: event.timestamp,
      })

      const deferred = new Map(state.deferredParentMessages)
      deferred.delete(forkId)
      return { ...state, deferredParentMessages: deferred }
    },

    fork_removed: ({ event, state, emit }) => {
      const existing = state.forks.get(event.forkId)

      if (!existing) {
        throw new Error(`[ForkProjection] Invalid state transition: fork_removed for unknown fork ${event.forkId}`)
      }

      if (existing.status !== 'completed') {
        throw new Error(`[ForkProjection] Invalid state transition: fork_removed for fork ${event.forkId} in state '${existing.status}' (expected 'completed')`)
      }

      emit.forkRemoved({
        forkId: event.forkId,
        parentForkId: event.parentForkId,
        name: existing.name
      })

      return state
    }
  }
}))