/**
 * CompactionProjection (Forked)
 *
 * Pure FSM + policy. Token budget tracking lives in WindowProjection.
 * This projection owns the compaction lifecycle state and derives
 * shouldCompact / contextLimitBlocked from Window's tokenEstimate.
 */

import { Schema } from 'effect'
import { Projection, Signal } from '@magnitudedev/event-core'
import { FSM } from '@magnitudedev/utils'
import { CompletedTurnSchema, type CompletedTurn } from '../window/types'
import { CompactResultSchema } from '../compaction/context'
const { defineFSM } = FSM

import type { AppEvent, SessionContext, CompactionOutcome } from '../events'
import { SessionContextSchema } from './session-context'
import { compactionSignals } from './compaction-signals'
import { AgentRoutingProjection } from './agent-routing'
import { AgentLifecycleProjection, type AgentLifecycleState } from './agent-lifecycle'
import { WindowProjection } from '../window'
import { ConfigAmbient, getSlotConfigForRole, type SlotConfig, type ConfigState } from '../ambient/config-ambient'
import { getForkInfo } from '../agents/registry'


// =============================================================================
// Policy Helpers
// =============================================================================

function isCompactionBlocking(tag: CompactionState['_tag']): boolean {
  return tag !== 'idle'
}

function deriveShouldCompact(
  tag: CompactionState['_tag'],
  tokenEstimate: number,
  limits: SlotConfig
): boolean {
  return tag === 'idle' && tokenEstimate > limits.softCap
}

function computeContextLimitBlocked(
  tag: CompactionState['_tag'],
  tokenEstimate: number,
  limits: SlotConfig
): boolean {
  return isCompactionBlocking(tag) && tokenEstimate >= limits.hardCap
}


// =============================================================================
// FSM State
// =============================================================================

interface AmbientCompactionFields {
  readonly contextLimitBlocked: boolean
  readonly shouldCompact: boolean
}

const AmbientCompactionFieldsSchema = {
  contextLimitBlocked: Schema.Boolean,
  shouldCompact: Schema.Boolean,
} as const

export const CompactionOutcomeSchema = Schema.Union(
  Schema.Struct({ isFallback: Schema.Literal(true) }),
  Schema.Struct({ isFallback: Schema.Literal(false), compactResult: CompactResultSchema }),
)

export class CompactionIdle extends Schema.TaggedClass<CompactionIdle>()('idle', AmbientCompactionFieldsSchema) {}

export class Compacting extends Schema.TaggedClass<Compacting>()('compacting', {
  ...AmbientCompactionFieldsSchema,
  compactedMessageCount: Schema.Number,
}) {}

export class PendingInjection extends Schema.TaggedClass<PendingInjection>()('pendingInjection', {
  ...AmbientCompactionFieldsSchema,
  turn: CompletedTurnSchema,
  compactionOutcome: CompactionOutcomeSchema,
  compactedMessageCount: Schema.Number,
  inputTokens: Schema.NullOr(Schema.Number),
  outputTokens: Schema.NullOr(Schema.Number),
  refreshedContext: Schema.NullOr(SessionContextSchema),
}) {}

export const CompactionLifecycle = defineFSM(
  { idle: CompactionIdle, compacting: Compacting, pendingInjection: PendingInjection },
  { idle: ['compacting'], compacting: ['pendingInjection', 'idle'], pendingInjection: ['idle'] }
)

export const CompactionStateSchema = Schema.Union(CompactionIdle, Compacting, PendingInjection)
export type CompactionState = typeof CompactionStateSchema.Type

function emitLifecycleSignals(
  oldState: CompactionState,
  newState: CompactionState,
  forkId: string | null,
  emit: {
    readonly shouldCompactChanged: (value: { forkId: string | null; shouldCompact: boolean }) => void
    readonly compactionBlockingChanged: (value: { forkId: string | null; blocking: boolean }) => void
    readonly contextLimitBlockedChanged: (value: { forkId: string | null; blocked: boolean }) => void
  }
): void {
  if (oldState.shouldCompact !== newState.shouldCompact) {
    emit.shouldCompactChanged({ forkId, shouldCompact: newState.shouldCompact })
  }

  const oldBlocking = isCompactionBlocking(oldState._tag)
  const newBlocking = isCompactionBlocking(newState._tag)
  if (oldBlocking !== newBlocking) {
    emit.compactionBlockingChanged({ forkId, blocking: newBlocking })
  }

  if (oldState.contextLimitBlocked !== newState.contextLimitBlocked) {
    emit.contextLimitBlockedChanged({ forkId, blocked: newState.contextLimitBlocked })
  }
}

function withAmbient(
  state: CompactionState,
  updates: Partial<AmbientCompactionFields>
): CompactionState {
  return CompactionLifecycle.hold(state, updates)
}

function getForkConfig(
  configState: ConfigState,
  agentStatus: AgentLifecycleState,
  forkId: string | null,
): SlotConfig | null {
  const info = getForkInfo(agentStatus, forkId)
  if (!info) return null
  const roleId = info.roleId
  if (!roleId) return null
  return getSlotConfigForRole(configState, roleId)
}

function recomputePolicy(
  fork: CompactionState,
  tokenEstimate: number,
  limits: SlotConfig,
): CompactionState {
  // During active compaction, preserve contextLimitBlocked so compaction_failed
  // can determine retry intent. isCompactionBlocking(_tag) governs system
  // blocking during compaction; contextLimitBlocked is only actionable when idle.
  const contextLimitBlocked = fork._tag === 'idle'
    ? computeContextLimitBlocked(fork._tag, tokenEstimate, limits)
    : fork.contextLimitBlocked

  return withAmbient(fork, {
    shouldCompact: deriveShouldCompact(fork._tag, tokenEstimate, limits),
    contextLimitBlocked,
  })
}

// =============================================================================
// Projection
// =============================================================================

export const CompactionProjection = Projection.defineForked<AppEvent>()({
  name: 'Compaction',
  forkState: CompactionStateSchema,

  reads: [AgentRoutingProjection, AgentLifecycleProjection, WindowProjection] as const,
  ambients: [ConfigAmbient] as const,

  signals: compactionSignals,

  initialFork: new CompactionIdle({
    contextLimitBlocked: false,
    shouldCompact: false,
  }),

  eventHandlers: {
    compaction_started: ({ event, fork, emit }) => {
      if (fork._tag !== 'idle') return fork

      // Preserve contextLimitBlocked through the transition so compaction_failed
      // can determine whether retry is needed. During compaction, isCompactionBlocking(_tag)
      // governs system blocking — contextLimitBlocked is only actionable when idle.
      const nextState = CompactionLifecycle.transition(fork, 'compacting', {
        compactedMessageCount: event.compactedMessageCount,
        shouldCompact: false,
        contextLimitBlocked: fork.contextLimitBlocked,
      })

      emitLifecycleSignals(fork, nextState, event.forkId, emit)
      return nextState
    },

    compaction_prepared: ({ event, fork, emit }) => {
      if (fork._tag !== 'compacting') return fork

      // Preserve contextLimitBlocked through the transition for the same reason
      // as compaction_started — compaction_failed needs it to determine retry intent.
      const compactionOutcome: CompactionOutcome = event.isFallback
        ? { isFallback: true }
        : { isFallback: false, compactResult: event.compactResult }
      const nextState = CompactionLifecycle.transition(fork, 'pendingInjection', {
        turn: event.turn,
        compactionOutcome,
        compactedMessageCount: event.compactedMessageCount,
        inputTokens: event.inputTokens,
        outputTokens: event.outputTokens,
        refreshedContext: event.refreshedContext,
        shouldCompact: false,
        contextLimitBlocked: fork.contextLimitBlocked,
      })

      emitLifecycleSignals(fork, nextState, event.forkId, emit)
      return nextState
    },

    compaction_injected: ({ event, fork, emit, ambient, read }) => {
      if (fork._tag !== 'pendingInjection') return fork

      // Emit the compactionInjected signal with all data from PendingInjection
      emit.compactionInjected({
        forkId: event.forkId,
        turn: fork.turn,
        compactionOutcome: fork.compactionOutcome,
        compactedMessageCount: fork.compactedMessageCount,
        inputTokens: fork.inputTokens,
        outputTokens: fork.outputTokens,
        refreshedContext: fork.refreshedContext,
      })

      // Transition to idle with shouldCompact: false.
      // The window rewrite hasn't happened yet (it fires via signal handler after this),
      // so reading WindowProjection here would see stale pre-compaction token estimates.
      // The subsequent tokenEstimateChanged signal from WindowProjection will recompute policy.
      const nextState = CompactionLifecycle.transition(fork, 'idle', {
        shouldCompact: false,
        contextLimitBlocked: false,
      })

      emitLifecycleSignals(fork, nextState, event.forkId, emit)
      return nextState
    },

    compaction_failed: ({ event, fork, emit }) => {
      // Window is unchanged after failure — no tokenEstimateChanged signal will fire.
      // Preserve retry intent: if we were under pressure (contextLimitBlocked), keep shouldCompact true.
      const wasBlocked = fork.contextLimitBlocked

      if (fork._tag === 'idle') {
        if (!fork.contextLimitBlocked) return fork
        const nextState = withAmbient(fork, {
          contextLimitBlocked: false,
          shouldCompact: wasBlocked,
        })
        emitLifecycleSignals(fork, nextState, event.forkId, emit)
        return nextState
      }

      const nextState = CompactionLifecycle.transition(fork, 'idle', {
        shouldCompact: wasBlocked,
        contextLimitBlocked: false,
      })

      emitLifecycleSignals(fork, nextState, event.forkId, emit)
      return nextState
    },

    interrupt: ({ event, fork, emit }) => {
      if (fork._tag === 'idle') {
        if (!fork.contextLimitBlocked) return fork
        const nextState = withAmbient(fork, {
          contextLimitBlocked: false,
          shouldCompact: false,
        })
        emitLifecycleSignals(fork, nextState, event.forkId, emit)
        return nextState
      }

      const nextState = CompactionLifecycle.transition(fork, 'idle', {
        shouldCompact: false,
        contextLimitBlocked: false,
      })

      emitLifecycleSignals(fork, nextState, event.forkId, emit)
      return nextState
    },

    context_limit_hit: ({ event, fork, emit }) => {
      const nextState = withAmbient(fork, {
        contextLimitBlocked: true,
        shouldCompact: fork._tag === 'idle' ? true : fork.shouldCompact,
      })

      emitLifecycleSignals(fork, nextState, event.forkId, emit)
      return nextState
    },
  },

  signalHandlers: (on) => [
    on(AgentRoutingProjection.signals.agentRegistered, ({ value, state }) => {
      const { forkId } = value

      const newForkState = new CompactionIdle({
        shouldCompact: false,
        contextLimitBlocked: false,
      })

      return {
        ...state,
        forks: new Map(state.forks).set(forkId, newForkState),
      }
    }),

    on(WindowProjection.signals.tokenEstimateChanged, ({ value, state, emit, ambient, read }) => {
      const fork = state.forks.get(value.forkId)
      if (!fork) return state

      const configState = ambient.get(ConfigAmbient)
      const agentStatus = read(AgentLifecycleProjection)
      const limits = getForkConfig(configState, agentStatus, value.forkId)
      if (!limits) return state

      const nextState = recomputePolicy(fork, value.tokenEstimate, limits)
      if (nextState === fork) return state

      emitLifecycleSignals(fork, nextState, value.forkId, emit)
      return {
        ...state,
        forks: new Map(state.forks).set(value.forkId, nextState),
      }
    }),
  ],

  ambientHandlers: (on) => ([
    on(ConfigAmbient, ({ value, state, emit, read }) => {
      const nextForks = new Map<string | null, CompactionState>()
      const agentStatus = read(AgentLifecycleProjection)

      const windowForks = read(WindowProjection)

      for (const [forkId, fork] of state.forks) {
        const limits = getForkConfig(value, agentStatus, forkId)
        if (!limits) {
          nextForks.set(forkId, fork)
          continue
        }
        const windowFork = windowForks.forks.get(forkId)
        const tokenEstimate = windowFork?.tokenEstimate ?? 0
        const nextFork = recomputePolicy(fork, tokenEstimate, limits)
        emitLifecycleSignals(fork, nextFork, forkId, emit)
        nextForks.set(forkId, nextFork)
      }

      return {
        ...state,
        forks: nextForks,
      }
    }),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ] as any),
})
