/**
 * TurnProjection (Forked)
 *
 * Turn scheduling + lifecycle tracking, per-fork.
 * Each fork has independent lifecycle, trigger queue, and inbound communication buffer.
 */

import { Projection, Signal } from '@magnitudedev/event-core'
import { FSM } from '@magnitudedev/event-core'
const { defineFSM } = FSM
import { Data } from 'effect'
import { logger } from '@magnitudedev/logger'
import type { AppEvent, TurnCompleted } from '../events'
import type { ToolKey } from '../catalog'
import type { XmlToolResult } from '@magnitudedev/xml-act'
import { AgentRoutingProjection } from './agent-routing'
import { UserMessageResolutionProjection } from './user-message-resolution'
import { CompactionProjection } from './compaction'
import { WorkflowProjection } from './workflow'
import { createId } from '../util/id'

// =============================================================================
// Types
// =============================================================================

export interface ToolCall {
  readonly toolCallId: string
  // Internal turn bookkeeping stays on catalog keys. Model-facing rendering resolves
  // the XML tag separately at the inbox/memory boundary.
  readonly toolKey: ToolKey
  readonly input: unknown
  readonly result?: XmlToolResult
}

export type TurnTrigger =
  | { readonly _tag: 'communication' }
  | { readonly _tag: 'chain_continue'; readonly chainId: string }
  | { readonly _tag: 'subagent_completed'; readonly agentId: string; readonly turnId: string }
  | { readonly _tag: 'wake' }
  | { readonly _tag: 'skill_started'; readonly skillName: string }
  | { readonly _tag: 'phase_verdict' }
  | { readonly _tag: 'oneshot' }
  | { readonly _tag: 'agent_created'; readonly agentId: string }

export interface PendingInboundCommunication {
  readonly id: string
  readonly source: 'agent' | 'user'
  readonly replyPolicy: 'parent_default' | 'user_reply_once'
  readonly direction: 'from_agent' | 'to_agent'
  readonly agentId: string
  readonly agentName?: string
  readonly agentRole?: string
  readonly forkId: string | null
  readonly content: string
  readonly preview: string
  readonly timestamp: number
  readonly arrivedAtTurnId: string | null
  readonly readAtTurnId?: string
}

interface TurnAmbient {
  readonly completedTurns: number
  readonly triggers: readonly TurnTrigger[]
  readonly pendingInboundCommunications: readonly PendingInboundCommunication[]
  readonly softInterrupted: boolean
  readonly parentForkId: string | null
}

export class TurnIdle extends Data.TaggedClass('idle')<TurnAmbient> {}

export class TurnActive extends Data.TaggedClass('active')<
  TurnAmbient & {
    readonly turnId: string
    readonly chainId: string
    readonly toolCalls: readonly ToolCall[]
    readonly currentTurnAllowsDirectUserReply: boolean
  }
> {}

export class TurnInterrupting extends Data.TaggedClass('interrupting')<
  TurnAmbient & {
    readonly turnId: string
    readonly chainId: string
    readonly toolCalls: readonly ToolCall[]
    readonly currentTurnAllowsDirectUserReply: boolean
  }
> {}

export const TurnLifecycle = defineFSM(
  { idle: TurnIdle, active: TurnActive, interrupting: TurnInterrupting },
  { idle: ['active'], active: ['idle', 'interrupting'], interrupting: ['idle'] }
)

export type TurnLifecycleState = TurnIdle | TurnActive | TurnInterrupting
export type ForkTurnState = TurnLifecycleState

type TurnTerminationReason = 'completed' | 'cancelled' | 'error'

function toPreview(text: string): string {
  const normalized = text.replace(/\s+/g, ' ').trim()
  if (normalized.length <= 120) return normalized
  return normalized.slice(0, 117) + '...'
}

function extractTextFromParts(parts: readonly { readonly type: string; readonly text?: string }[]): string {
  return parts
    .filter((part) => part.type === 'text')
    .map((part) => part.text ?? '')
    .join('')
}

function isStable(fork: TurnLifecycleState): boolean {
  return fork._tag === 'idle' && fork.triggers.length === 0 && !fork.softInterrupted
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

// =============================================================================
// Projection
// =============================================================================

export const TurnProjection = Projection.defineForked<AppEvent, TurnLifecycleState>()({
  name: 'Turn',

  reads: [
    AgentRoutingProjection,
    UserMessageResolutionProjection,
    CompactionProjection,
    WorkflowProjection,
  ] as const,

  signals: {
    turnActivated: Signal.create<{ forkId: string | null; turnId: string; chainId: string }>('Turn/turnActivated'),
    turnInterrupting: Signal.create<{ forkId: string | null; turnId: string }>('Turn/turnInterrupting'),
    turnTerminated: Signal.create<{
      forkId: string | null
      turnId: string
      reason: TurnTerminationReason
      result?: TurnCompleted['result']
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
    softInterrupted: false,
    parentForkId: null,
  }),

  eventHandlers: {
    soft_interrupt: ({ fork }) =>
      TurnLifecycle.hold(fork, {
        softInterrupted: true,
      }),

    interrupt: ({ event, fork, emit }) => {
      const afterClear = clearTriggers(fork)

      if (afterClear._tag === 'active') {
        emit.turnInterrupting({
          forkId: event.forkId,
          turnId: afterClear.turnId,
        })

        return TurnLifecycle.transition(afterClear, 'interrupting', {
          softInterrupted: false,
          currentTurnAllowsDirectUserReply: afterClear.currentTurnAllowsDirectUserReply,
        })
      }

      return TurnLifecycle.hold(afterClear, {
        softInterrupted: false,
      })
    },

    oneshot_task: ({ fork }) => {
      const next = enqueueTrigger(fork, { _tag: 'oneshot' })
      return next
    },

    skill_started: ({ event, fork }) => {
      if (event.source !== 'user') return fork
      const next = enqueueTrigger(fork, { _tag: 'skill_started', skillName: event.skill.name })
      return next
    },

    phase_verdict: ({ fork }) => {
      const next = enqueueTrigger(fork, { _tag: 'phase_verdict' })
      return next
    },

    wake: ({ fork }) => {
      const next = enqueueTrigger(fork, { _tag: 'wake' })
      return next
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
        softInterrupted: false,
        currentTurnAllowsDirectUserReply: fork.pendingInboundCommunications.some(
          (message) => message.source === 'user' && message.replyPolicy === 'user_reply_once'
        ),
      })
    },

    tool_event: ({ event, fork }) => {
      if (fork._tag !== 'active') return fork
      if (fork.turnId !== event.turnId) return fork

      switch (event.event._tag) {
        case 'ToolInputStarted':
          return TurnLifecycle.hold(fork, {
            toolCalls: [
              ...fork.toolCalls,
              {
                toolCallId: event.toolCallId,
                toolKey: event.toolKey,
                input: undefined,
              },
            ],
          })

        case 'ToolInputReady': {
          const inner = event.event
          return TurnLifecycle.hold(fork, {
            toolCalls: fork.toolCalls.map((tc) =>
              tc.toolCallId === event.toolCallId ? { ...tc, input: inner.input } : tc
            ),
          })
        }

        case 'ToolExecutionEnded': {
          const inner = event.event
          return TurnLifecycle.hold(fork, {
            toolCalls: fork.toolCalls.map((tc) =>
              tc.toolCallId === event.toolCallId ? { ...tc, result: inner.result } : tc
            ),
          })
        }

        default:
          return fork
      }
    },

    turn_completed: ({ event, fork, emit }) => {
      if (fork._tag === 'idle') return fork
      if (fork.turnId !== event.turnId) return fork

      const turnWantsContinue = event.result.success ? event.result.turnDecision === 'continue' : !event.result.cancelled
      const shouldEnqueueContinue = turnWantsContinue && !fork.softInterrupted

      const nextTriggers = shouldEnqueueContinue
        ? [...fork.triggers, { _tag: 'chain_continue', chainId: fork.chainId } satisfies TurnTrigger]
        : fork.triggers

      emit.turnTerminated({
        forkId: event.forkId,
        turnId: event.turnId,
        reason: event.result.success ? 'completed' : event.result.cancelled ? 'cancelled' : 'error',
        result: event.result,
        triggersQueued: nextTriggers.length > 0,
      })

      // Always emit — lifecycle returning to idle is a readiness change

      return TurnLifecycle.transition(fork, 'idle', {
        completedTurns: fork.completedTurns + 1,
        triggers: nextTriggers,
        softInterrupted: false,
      })
    },

    turn_unexpected_error: ({ event, fork, emit }) => {
      if (fork._tag === 'idle') return fork
      if (fork.turnId !== event.turnId) return fork

      emit.turnTerminated({
        forkId: event.forkId,
        turnId: event.turnId,
        reason: 'error',
        triggersQueued: fork.triggers.length > 0,
      })


      return TurnLifecycle.transition(fork, 'idle', {
        completedTurns: fork.completedTurns + 1,
        softInterrupted: false,
      })
    },
  },

  globalEventHandlers: {
    turn_completed: ({ event, state }) => {
      if (event.forkId === null) return state

      const subFork = state.forks.get(event.forkId)
      if (!subFork) return state
      if (!isStable(subFork)) return state

      const parentId = subFork.parentForkId

      const parentFork = state.forks.get(parentId)
      if (!parentFork) return state

      const nextParent = enqueueTrigger(parentFork, { _tag: 'wake' })
      return {
        ...state,
        forks: new Map(state.forks).set(parentId, nextParent),
      }
    },

    turn_unexpected_error: ({ event, state }) => {
      if (event.forkId === null) return state

      const subFork = state.forks.get(event.forkId)
      if (!subFork) return state

      const parentId = subFork.parentForkId

      const parentFork = state.forks.get(parentId)
      if (!parentFork) return state

      const nextParent = enqueueTrigger(parentFork, { _tag: 'wake' })
      return {
        ...state,
        forks: new Map(state.forks).set(parentId, nextParent),
      }
    },

    subagent_user_killed: ({ event, state }) => {
      const parentId = event.parentForkId

      // Only wake parent if the killed subagent was NOT already idle/stable
      const subFork = event.forkId != null ? state.forks.get(event.forkId) : undefined
      if (subFork && isStable(subFork)) return state

      const parentFork = state.forks.get(parentId)
      if (!parentFork) return state

      const nextParent = enqueueTrigger(parentFork, { _tag: 'wake' })
      return {
        ...state,
        forks: new Map(state.forks).set(parentId, nextParent),
      }
    },
  },

  signalHandlers: (on) => [
    on(UserMessageResolutionProjection.signals.userMessageResolved, ({ value, state, emit }) => {
      const forkId = value.forkId
      const fork = state.forks.get(forkId)
      if (!fork) return state

      const contentText = extractTextFromParts(value.content)
      const next = TurnLifecycle.hold(fork, {
        triggers: [...fork.triggers, { _tag: 'communication' }],
        pendingInboundCommunications: [
          ...fork.pendingInboundCommunications,
          {
            id: createId(),
            source: 'user',
            replyPolicy: 'user_reply_once',
            direction: 'from_agent',
            agentId: 'user',
            forkId,
            content: contentText,
            preview: toPreview(contentText),
            timestamp: value.timestamp,
            arrivedAtTurnId: fork._tag === 'idle' ? null : fork.turnId,
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
            replyPolicy: 'parent_default',
            direction: 'from_agent',
            agentId: value.agentId,
            forkId,
            content: value.message,
            preview: toPreview(value.message),
            timestamp: value.timestamp,
            arrivedAtTurnId: fork._tag === 'idle' ? null : fork.turnId,
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
            replyPolicy: 'parent_default',
            direction: 'from_agent',
            agentId: value.agentId,
            forkId,
            content: value.message,
            preview: toPreview(value.message),
            timestamp: value.timestamp,
            arrivedAtTurnId: fork._tag === 'idle' ? null : fork.turnId,
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
