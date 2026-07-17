/**
 * TurnProjection (Forked)
 *
 * Turn scheduling + lifecycle tracking, per-fork.
 * Each fork has independent lifecycle, trigger queue, and inbound communication buffer.
 */

import { Projection, Signal } from '@magnitudedev/event-core'
import { FSM } from '@magnitudedev/utils'
const { defineFSM } = FSM
import { Schema, Option } from 'effect'
import { logger } from '@magnitudedev/logger'
import { outcomeWillChainContinue } from '../events'
import type { AppEvent, TurnOutcomeEvent } from '../events'
import { computeDelayMs, getRetryAfterHint } from '../util/retry-backoff'
import { JsonValueSchema } from '@magnitudedev/ai'
import { AgentLifecycleProjection, hasActiveWorkers } from './agent-lifecycle'
import { AgentRoutingProjection } from './agent-routing'
import { UserMessageResolutionProjection } from './user-message-resolution'
import { GoalProjection } from './goal'
import { createId } from '../util/id'

const ToolResultSchema = Schema.Union(
  Schema.TaggedStruct('Success', { output: JsonValueSchema }),
  Schema.TaggedStruct('Error', { error: Schema.Struct({ message: Schema.String }) }),
  Schema.TaggedStruct('Denied', { denial: JsonValueSchema }),
  Schema.TaggedStruct('Interrupted', {}),
  Schema.TaggedStruct('InputRejected', {
    issue: Schema.Struct({
      path: Schema.Array(Schema.Union(Schema.String, Schema.Number)),
      message: Schema.String,
    }),
    partialInput: JsonValueSchema,
  }),
)

export const ToolCallSchema = Schema.Struct({
  toolCallId: Schema.String,
  toolKey: Schema.String,
  input: JsonValueSchema,
  result: Schema.optionalWith(ToolResultSchema, { as: 'Option', exact: true }),
})
export type ToolCall = typeof ToolCallSchema.Type

export const TurnTriggerSchema = Schema.Union(
  Schema.TaggedStruct('communication', {}),
  Schema.TaggedStruct('chain_continue', { chainId: Schema.String, notBefore: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }) }),
  Schema.TaggedStruct('subagent_completed', { agentId: Schema.String, turnId: Schema.String }),
  Schema.TaggedStruct('wake', {}),
  Schema.TaggedStruct('agent_created', { agentId: Schema.String }),
)
export type TurnTrigger = typeof TurnTriggerSchema.Type

export const PendingInboundCommunicationSchema = Schema.Struct({
  id: Schema.String,
  source: Schema.Literal('agent', 'user'),
  direction: Schema.Literal('from_agent', 'to_agent'),
  agentId: Schema.String,
  agentName: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  agentRole: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  forkId: Schema.NullOr(Schema.String),
  content: Schema.String,
  preview: Schema.String,
  timestamp: Schema.Number,
  arrivedAtTurnId: Schema.NullOr(Schema.String),
  readAtTurnId: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  requiresAdvisor: Schema.optionalWith(Schema.Literal(true), { as: 'Option', exact: true }),
})
export type PendingInboundCommunication = typeof PendingInboundCommunicationSchema.Type

const TurnBaseFields = {
  completedTurns: Schema.Number,
  triggers: Schema.Array(TurnTriggerSchema),
  pendingInboundCommunications: Schema.Array(PendingInboundCommunicationSchema),
  parentForkId: Schema.NullOr(Schema.String),
  connectionRetryCount: Schema.Number,
} as const

export class TurnIdle extends Schema.TaggedClass<TurnIdle>()('idle', TurnBaseFields) {}

const RunningTurnFields = {
  ...TurnBaseFields,
  turnId: Schema.String,
  chainId: Schema.String,
  toolCalls: Schema.Array(ToolCallSchema),
  triggeredByUser: Schema.Boolean,
  requiresAdvisor: Schema.Boolean,
} as const

export class TurnActive extends Schema.TaggedClass<TurnActive>()('active', RunningTurnFields) {}

export class TurnInterrupting extends Schema.TaggedClass<TurnInterrupting>()('interrupting', RunningTurnFields) {}

export class TurnWaitingForUser extends Schema.TaggedClass<TurnWaitingForUser>()('waiting_for_user', TurnBaseFields) {}

export const TurnLifecycle = defineFSM(
  { idle: TurnIdle, active: TurnActive, interrupting: TurnInterrupting, waiting_for_user: TurnWaitingForUser },
  { idle: ['active', 'waiting_for_user'], active: ['idle', 'interrupting'], interrupting: ['idle', 'waiting_for_user'], waiting_for_user: ['idle'] }
)

export const TurnLifecycleStateSchema = Schema.Union(TurnIdle, TurnActive, TurnInterrupting, TurnWaitingForUser)
export type TurnLifecycleState = typeof TurnLifecycleStateSchema.Type
export type ForkTurnState = TurnLifecycleState

type TurnTerminationReason = 'completed' | 'cancelled' | 'error'

function toPreview(text: string): string {
  const normalized = text.replace(/\s+/g, ' ').trim()
  if (normalized.length <= 120) return normalized
  return normalized.slice(0, 117) + '...'
}

function clearTriggers(fork: TurnLifecycleState): TurnLifecycleState {
  return TurnLifecycle.hold(fork, {
    triggers: [],
  })
}

function enqueueTrigger(fork: TurnLifecycleState, trigger: TurnTrigger): TurnLifecycleState {
  return TurnLifecycle.hold(fork, {
    triggers: [...fork.triggers, trigger],
  })
}

function ensureCommunicationTrigger(triggers: readonly TurnTrigger[]): readonly TurnTrigger[] {
  return triggers.some((trigger) => trigger._tag === 'communication')
    ? triggers
    : [...triggers, { _tag: 'communication' }]
}

export function hasPendingAdvisorRequirement(fork: ForkTurnState | undefined): boolean {
  // TEMPORARILY DISABLED: advisor-required turn claims.
  // return fork?.pendingInboundCommunications.some((message) => message.requiresAdvisor === true) ?? false
  return false
}

export function turnRequiresAdvisor(fork: ForkTurnState | undefined, turnId: string): boolean {
  // TEMPORARILY DISABLED: forced advisor calls.
  // return fork?._tag === 'active' && fork.turnId === turnId && fork.requiresAdvisor
  return false
}

function enqueueInboundCommunication(
  fork: TurnLifecycleState,
  message: PendingInboundCommunication,
): TurnLifecycleState {
  const pendingInboundCommunications = [...fork.pendingInboundCommunications, message]

  return TurnLifecycle.hold(fork, {
    pendingInboundCommunications,
  })
}

// =============================================================================
// Projection
// =============================================================================

export const TurnProjection = Projection.defineForked<AppEvent>()({
  name: 'Turn',
  forkState: TurnLifecycleStateSchema,

  reads: [
    AgentLifecycleProjection,
    AgentRoutingProjection,
    UserMessageResolutionProjection,
    GoalProjection,
  ] as const,

  signals: {
    turnActivated: Signal.create<{ forkId: string | null; turnId: string; chainId: string }>('Turn/turnActivated'),
    turnInterrupting: Signal.create<{ forkId: string | null; turnId: string }>('Turn/turnInterrupting'),
    turnTerminated: Signal.create<{
      forkId: string | null
      turnId: string
      reason: TurnTerminationReason
      result?: TurnOutcomeEvent['outcome']
      triggersQueued: boolean
    }>('Turn/turnTerminated'),
    pendingInboundCommunicationsRead: Signal.create<{
      forkId: string | null
      turnId: string
      messages: readonly PendingInboundCommunication[]
      timestamp: number
    }>('Turn/pendingInboundCommunicationsRead'),
  },

  initialFork: new TurnIdle({
    completedTurns: 0,
    triggers: [],
    pendingInboundCommunications: [],
    parentForkId: null,
    connectionRetryCount: 0,
  }),

  eventHandlers: {
    interrupt: ({ event, fork, emit }) => {
      const isRoot = event.forkId === null
      const advisorRequirementPending = isRoot && hasPendingAdvisorRequirement(fork)
      const afterClear = clearTriggers(fork)

      if (advisorRequirementPending) {
        if (afterClear._tag === 'waiting_for_user') return afterClear
        if (afterClear._tag === 'interrupting') return afterClear
      }

      // Idempotent: already waiting_for_user
      if (afterClear._tag === 'waiting_for_user') return afterClear

      if (afterClear._tag === 'active') {
        emit.turnInterrupting({
          forkId: event.forkId,
          turnId: afterClear.turnId,
        })

        return TurnLifecycle.transition(afterClear, 'interrupting', {
          triggeredByUser: afterClear.triggeredByUser,
        })
      }

      // Root was idle → go directly to waiting_for_user
      if (isRoot) {
        return TurnLifecycle.transition(afterClear, 'waiting_for_user', {})
      }

      // Worker was idle → stay idle
      return afterClear
    },

    wake: ({ fork }) => {
      const next = enqueueTrigger(fork, { _tag: 'wake' })
      return next
    },

    goal_started: ({ event, fork }) => {
      if (event.forkId !== null) return fork
      const triggers = fork.triggers.some((trigger) => trigger._tag === 'wake')
        ? fork.triggers
        : [...fork.triggers, { _tag: 'wake' } satisfies TurnTrigger]

      if (fork._tag === 'waiting_for_user') {
        return TurnLifecycle.transition(fork, 'idle', { triggers })
      }

      return TurnLifecycle.hold(fork, { triggers })
    },

    agent_created: ({ event, fork }) => {
      const withParent = TurnLifecycle.hold(fork, { parentForkId: event.parentForkId })
      if (event.message === null) {
        return withParent
      }
      const next = enqueueTrigger(withParent, { _tag: 'agent_created', agentId: event.agentId })
      return next
    },

    turn_started: ({ event, fork, emit }) => {
      if (fork._tag !== 'idle') {
        logger.error(`[TurnProjection] Invalid turn_started while ${fork._tag} on fork ${event.forkId ?? 'root'}`)
        return fork
      }

      const requiresAdvisor = event.forkId === null && hasPendingAdvisorRequirement(fork)

      if (fork.pendingInboundCommunications.length > 0) {
        emit.pendingInboundCommunicationsRead({
          forkId: event.forkId,
          turnId: event.turnId,
          messages: fork.pendingInboundCommunications,
          timestamp: event.timestamp,
        })
      }

      emit.turnActivated({
        forkId: event.forkId,
        turnId: event.turnId,
        chainId: event.chainId,
      })

      return TurnLifecycle.transition(fork, 'active', {
        turnId: event.turnId,
        chainId: event.chainId,
        toolCalls: [],
        triggers: [],
        pendingInboundCommunications: [],
        triggeredByUser: fork.pendingInboundCommunications.some(
          (message) => message.source === 'user'
        ),
        requiresAdvisor,
      })
    },

    turn_outcome: ({ event, fork, emit, read }) => {
      if (fork._tag === 'idle' || fork._tag === 'waiting_for_user') return fork
      if (fork.turnId !== event.turnId) return fork

      const isRoot = event.forkId === null
      const isUserInterruptedRoot =
        isRoot &&
        fork._tag === 'interrupting' &&
        event.outcome._tag === 'Cancelled' &&
        event.outcome.reason._tag === 'UserInterrupt'

      const shouldEnqueueContinue = outcomeWillChainContinue(event.outcome)
      const isConnectionFailure = event.outcome._tag === 'ConnectionFailure'
      const advisorRequirementPending = hasPendingAdvisorRequirement(fork)

      // Increment retry count on ConnectionFailure, reset on anything else.
      // Cortex enforces the cap by transforming the outcome before publishing,
      // so the projection trusts what it sees here.
      const nextRetryCount = isConnectionFailure ? fork.connectionRetryCount + 1 : 0

      // For connection-failure retries, schedule the chain_continue with a
      // notBefore timestamp computed from the retry count and any server hint.
      const notBefore =
        shouldEnqueueContinue && isConnectionFailure
          ? event.timestamp + computeDelayMs(fork.connectionRetryCount, getRetryAfterHint(event.outcome))
          : undefined

      const goalState = isRoot ? read(GoalProjection) : null
      const agentStatus = isRoot ? read(AgentLifecycleProjection) : null
      const isUserInterrupt = event.outcome._tag === 'Cancelled' && event.outcome.reason._tag === 'UserInterrupt'
      const shouldEnqueueGoalReminder =
        isRoot &&
        !shouldEnqueueContinue &&
        !isUserInterrupt &&
        goalState?.active != null &&
        agentStatus !== null &&
        !hasActiveWorkers(agentStatus)

      const nextTriggers = shouldEnqueueContinue
        ? [...fork.triggers, { _tag: 'chain_continue', chainId: fork.chainId, notBefore: notBefore !== undefined ? Option.some(notBefore) : Option.none() } satisfies TurnTrigger]
        : shouldEnqueueGoalReminder
          ? [...fork.triggers, { _tag: 'wake' } satisfies TurnTrigger]
        : fork.triggers

      emit.turnTerminated({
        forkId: event.forkId,
        turnId: event.turnId,
        reason:
          event.outcome._tag === 'Cancelled'
            ? 'cancelled'
            : event.outcome._tag === 'Completed'
              ? 'completed'
              : 'error',
        result: event.outcome,
        triggersQueued: isUserInterruptedRoot ? false : nextTriggers.length > 0,
      })

      if (isUserInterruptedRoot) {
        return TurnLifecycle.transition(fork, 'waiting_for_user', {
          completedTurns: fork.completedTurns + 1,
          triggers: [],
          connectionRetryCount: 0,
        })
      }

      // Standard path: all worker outcomes, root normal outcomes, root crash recovery
      return TurnLifecycle.transition(fork, 'idle', {
        completedTurns: fork.completedTurns + 1,
        triggers: nextTriggers,
        connectionRetryCount: nextRetryCount,
      })
    },

    shell_completed: ({ event, fork }) => {
      if (fork._tag === 'interrupting') return fork
      if (fork._tag === 'idle' || fork._tag === 'waiting_for_user') {
        return enqueueTrigger(fork, { _tag: 'wake' })
      }
      return fork
    },
  },

  globalEventHandlers: {
    turn_outcome: ({ event, state }) => {
      if (event.forkId === null) return state

      const subFork = state.forks.get(event.forkId)
      if (!subFork) return state
      if (subFork._tag !== 'idle' || subFork.triggers.length > 0) return state

      // Don't wake parent for user-killed workers — the coordinator already knows
      const isUserKilled = event.outcome._tag === 'Cancelled' && event.outcome.reason._tag === 'UserInterrupt'
      if (isUserKilled) return state

      const parentId = subFork.parentForkId

      const parentFork = state.forks.get(parentId)
      if (!parentFork) return state

      const nextParent = enqueueTrigger(parentFork, { _tag: 'wake' })
      return {
        ...state,
        forks: new Map(state.forks).set(parentId, nextParent),
      }
    },

    worker_user_killed: ({ event, state }) => {
      const parentId = event.parentForkId

      // Only wake coordinator if the killed subagent was NOT already idle with no triggers
      const subFork = event.forkId != null ? state.forks.get(event.forkId) : undefined
      if (subFork && subFork._tag === 'idle' && subFork.triggers.length === 0) return state

      const parentFork = state.forks.get(parentId)
      if (!parentFork) return state

      const nextParent = enqueueTrigger(parentFork, { _tag: 'wake' })
      return {
        ...state,
        forks: new Map(state.forks).set(parentId, nextParent),
      }
    },

    observer_outcome: ({ event, state }) => {
      // TEMPORARILY DISABLED: Observer/Advisor escalation.
      // Historical observer_outcome events may still replay, but they should not
      // enqueue new advisor-required work while these features are off.
      return state

      /*
      if (!event.escalate) return state

      // Leader escalation: enqueue an advisor-required communication on root.
      if (event.forkId === null) {
        const rootFork = state.forks.get(null)
        if (!rootFork) return state

        const content = event.justification
          ? `System has detected ${event.justification}. Contact the advisor (message_advisor) for guidance.`
          : 'Observer recommends contacting the advisor for guidance.'

        const nextRoot = enqueueInboundCommunication(rootFork, {
          id: createId(),
          source: 'agent',
          direction: 'from_agent',
          agentId: 'observer',
          agentName: 'Observer',
          agentRole: 'observer',
          forkId: null,
          content,
          preview: toPreview(content),
          timestamp: Date.now(),
          arrivedAtTurnId: (rootFork._tag === 'idle' || rootFork._tag === 'waiting_for_user') ? null : rootFork.turnId,
          requiresAdvisor: true,
        })

        return { ...state, forks: new Map(state.forks).set(null, nextRoot) }
      }

      // Worker escalation -> target parent fork, source is observer
      const content = event.justification
        ? `System has detected ${event.justification}. Contact the advisor (message_advisor) for guidance.`
        : 'Observer recommends contacting the advisor for guidance.'

      const fork = state.forks.get(event.forkId)
      if (!fork) return state
      const parentId = fork.parentForkId
      if (parentId === null) return state

      const parentFork = state.forks.get(parentId)
      if (!parentFork) return state

      return {
        ...state,
        forks: new Map(state.forks).set(parentId, enqueueInboundCommunication(parentFork, {
          id: createId(),
          source: 'agent',
          direction: 'from_agent',
          agentId: 'observer',
          agentName: 'Observer',
          agentRole: 'observer',
          forkId: parentId,
          content,
          preview: toPreview(content),
          timestamp: Date.now(),
          arrivedAtTurnId: (parentFork._tag === 'idle' || parentFork._tag === 'waiting_for_user') ? null : parentFork.turnId,
        })),
      }
      */
    },
  },

  signalHandlers: (on) => [
    on(UserMessageResolutionProjection.signals.userMessageResolved, ({ value, state, emit }) => {
      const forkId = value.forkId
      const fork = state.forks.get(forkId)
      if (!fork) return state

      const contentText = value.text

      // Root is waiting_for_user → transition to idle with user trigger
      if (fork._tag === 'waiting_for_user') {
        const next = TurnLifecycle.transition(fork, 'idle', {
          triggers: [{ _tag: 'communication' }],
          pendingInboundCommunications: [
            ...fork.pendingInboundCommunications,
            {
              id: createId(),
              source: 'user',
              direction: 'from_agent',
              agentId: 'user',
              agentName: Option.none(),
              agentRole: Option.none(),
              forkId,
              content: contentText,
              preview: toPreview(contentText),
              timestamp: value.timestamp,
              arrivedAtTurnId: null,
              readAtTurnId: Option.none(),
              requiresAdvisor: Option.none(),
            },
          ],
        })
        return { ...state, forks: new Map(state.forks).set(forkId, next) }
      }

      // Normal path — append trigger and communication
      const next = TurnLifecycle.hold(fork, {
        triggers: [...fork.triggers, { _tag: 'communication' }],
        pendingInboundCommunications: [
          ...fork.pendingInboundCommunications,
          {
            id: createId(),
            source: 'user',
            direction: 'from_agent',
            agentId: 'user',
            agentName: Option.none(),
            agentRole: Option.none(),
            forkId,
            content: contentText,
            preview: toPreview(contentText),
            timestamp: value.timestamp,
            arrivedAtTurnId: fork._tag === 'idle' ? null : fork.turnId,
            readAtTurnId: Option.none(),
            requiresAdvisor: Option.none(),
          },
        ],
      })

      return {
        ...state,
        forks: new Map(state.forks).set(forkId, next),
      }
    }),

    on(AgentRoutingProjection.signals.agentResponse, ({ value, state, emit }) => {
      const forkId = value.targetForkId
      const fork = state.forks.get(forkId)
      if (!fork) return state

      const next = TurnLifecycle.hold(fork, {
        triggers: [
          ...fork.triggers,
          { _tag: 'communication' },
        ],
        pendingInboundCommunications: [
          ...fork.pendingInboundCommunications,
          {
            id: createId(),
            source: 'agent',
            direction: 'from_agent',
            agentId: value.agentId,
            agentName: Option.none(),
            agentRole: Option.none(),
            forkId,
            content: value.message,
            preview: toPreview(value.message),
            timestamp: value.timestamp,
            arrivedAtTurnId: (fork._tag === 'idle' || fork._tag === 'waiting_for_user') ? null : fork.turnId,
            readAtTurnId: Option.none(),
            requiresAdvisor: Option.none(),
          },
        ],
      })

      return {
        ...state,
        forks: new Map(state.forks).set(forkId, next),
      }
    }),

    on(AgentRoutingProjection.signals.agentMessage, ({ value, state, emit }) => {
      const forkId = value.targetForkId
      const fork = state.forks.get(forkId)
      if (!fork) return state

      const next = TurnLifecycle.hold(fork, {
        triggers: [
          ...fork.triggers,
          { _tag: 'communication' },
        ],
        pendingInboundCommunications: [
          ...fork.pendingInboundCommunications,
          {
            id: createId(),
            source: 'agent',
            direction: 'from_agent',
            agentId: value.agentId,
            agentName: Option.none(),
            agentRole: Option.none(),
            forkId,
            content: value.message,
            preview: toPreview(value.message),
            timestamp: value.timestamp,
            arrivedAtTurnId: (fork._tag === 'idle' || fork._tag === 'waiting_for_user') ? null : fork.turnId,
            readAtTurnId: Option.none(),
            requiresAdvisor: Option.none(),
          },
        ],
      })

      return {
        ...state,
        forks: new Map(state.forks).set(forkId, next),
      }
    }),
  ],
})
