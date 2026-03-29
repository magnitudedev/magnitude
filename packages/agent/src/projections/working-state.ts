/**
 * WorkingStateProjection (Forked)
 *
 * Core state machine controlling the turn loop, per-fork.
 * Each fork has independent working/willContinue state.
 */

import { Signal, Projection } from '@magnitudedev/event-core'
import { logger } from '@magnitudedev/logger'
import type { AppEvent } from '../events'
import { AgentRoutingProjection } from './agent-routing'
import { createId } from '../util/id'
import { CompactionProjection } from './compaction'
import { UserMessageResolutionProjection } from './user-message-resolution'

function toPreview(text: string): string {
  const normalized = text.replace(/\s+/g, ' ').trim()
  if (normalized.length <= 120) return normalized
  return normalized.slice(0, 117) + '...'
}


// =============================================================================
// State
// =============================================================================

/** Per-fork working state */
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

export interface ForkWorkingState {
  readonly parentForkId: string | null
  readonly working: boolean
  readonly willContinue: boolean
  readonly hasQueuedMessages: boolean
  readonly pendingWake: boolean
  readonly currentChainId: string | null
  readonly currentTurnId: string | null
  readonly compactionPending: boolean
  readonly contextLimitBlocked: boolean
  readonly pendingApproval: boolean
  readonly softInterrupted: boolean
  readonly currentTurnAllowsDirectUserReply: boolean
  readonly pendingSeeVerdict: boolean
  readonly pendingInboundCommunications: readonly PendingInboundCommunication[]
}

// =============================================================================
// Derived
// =============================================================================

export const shouldTrigger = (state: ForkWorkingState): boolean =>
  !state.working && state.willContinue && !state.compactionPending && !state.contextLimitBlocked

export const isStable = (state: ForkWorkingState): boolean =>
  !state.working && !state.willContinue && !state.compactionPending && !state.contextLimitBlocked

// =============================================================================
// Projection
// =============================================================================

export const WorkingStateProjection = Projection.defineForked<AppEvent, ForkWorkingState>()({
  name: 'WorkingState',

  reads: [AgentRoutingProjection, CompactionProjection, UserMessageResolutionProjection] as const,

  initialFork: {
    parentForkId: null,
    working: false,
    willContinue: false,
    hasQueuedMessages: false,
    pendingWake: false,
    currentChainId: null,
    currentTurnId: null,
    compactionPending: false,
    contextLimitBlocked: false,
    pendingApproval: false,
    softInterrupted: false,
    pendingSeeVerdict: false,
    currentTurnAllowsDirectUserReply: false,
    pendingInboundCommunications: []
  },

  signals: {
    shouldTriggerChanged: Signal.create<{ forkId: string | null; shouldTrigger: boolean; chainId: string | null }>('WorkingState/shouldTriggerChanged'),
    forkBecameStable: Signal.create<{ forkId: string | null; timestamp: number }>('WorkingState/forkBecameStable'),
    softInterruptResolved: Signal.create<{ forkId: string }>('WorkingState/softInterruptResolved'),
    pendingInboundCommunicationsRead: Signal.create<{ forkId: string | null; turnId: string; messages: readonly PendingInboundCommunication[]; timestamp: number }>('WorkingState/pendingInboundCommunicationsRead'),
  },

  eventHandlers: {
    oneshot_task: ({ event, fork, emit }) => {
      const newFork: ForkWorkingState = {
        ...fork,
        willContinue: true,
      }
      if (shouldTrigger(newFork) !== shouldTrigger(fork)) {
        emit.shouldTriggerChanged({
          forkId: event.forkId,
          shouldTrigger: shouldTrigger(newFork),
          chainId: newFork.currentChainId,
        })
      }
      return newFork
    },

    skill_activated: ({ event, fork }) => {
      return fork
    },

    skill_started: ({ event, fork, emit }) => {
      // Assistant-sourced skill_started happens mid-turn, no trigger needed
      if (event.source !== 'user') return fork

      const isQueued = fork.working
      const newFork: ForkWorkingState = {
        ...fork,
        willContinue: true,
        hasQueuedMessages: fork.hasQueuedMessages || isQueued,
      }

      if (shouldTrigger(newFork) !== shouldTrigger(fork)) {
        emit.shouldTriggerChanged({
          forkId: event.forkId,
          shouldTrigger: shouldTrigger(newFork),
          chainId: newFork.currentChainId
        })
      }
      return newFork
    },

    phase_verdict: ({ event, fork, emit }) => {
      if (fork.working) {
        // Verdict arrived mid-turn — defer until turn completes
        return { ...fork, pendingSeeVerdict: true }
      }

      const newFork: ForkWorkingState = {
        ...fork,
        willContinue: true,
      }

      if (shouldTrigger(newFork) !== shouldTrigger(fork)) {
        emit.shouldTriggerChanged({
          forkId: event.forkId,
          shouldTrigger: shouldTrigger(newFork),
          chainId: newFork.currentChainId
        })
      }

      return newFork
    },

    wake: ({ event, fork, emit }) => {
      const newFork: ForkWorkingState = {
        ...fork,
        pendingWake: true,
        willContinue: fork.working ? fork.willContinue : true,
      }

      if (shouldTrigger(newFork) !== shouldTrigger(fork)) {
        emit.shouldTriggerChanged({
          forkId: event.forkId,
          shouldTrigger: shouldTrigger(newFork),
          chainId: newFork.currentChainId
        })
      }

      return newFork
    },

    turn_started: ({ event, fork, emit }) => {
      if (fork.working) {
        logger.error(`[WorkingState] OVERLAPPING TURN DETECTED: turn_started(${event.turnId}) arrived while turn ${fork.currentTurnId} is still in-flight on fork ${event.forkId ?? 'root'}`)
      }

      if (fork.pendingInboundCommunications.length > 0) {
        emit.pendingInboundCommunicationsRead({
          forkId: event.forkId,
          turnId: event.turnId,
          messages: fork.pendingInboundCommunications,
          timestamp: event.timestamp,
        })
      }

      return {
        ...fork,
        working: true,
        willContinue: false,
        pendingWake: false,
        hasQueuedMessages: false,
        currentTurnAllowsDirectUserReply: fork.pendingInboundCommunications.some(
          (message) => message.source === 'user' && message.replyPolicy === 'user_reply_once'
        ),
        currentTurnId: event.turnId,
        currentChainId: event.chainId,
        pendingInboundCommunications: [],
      }
    },

    turn_completed: ({ event, fork, emit }) => {
      if (fork.currentTurnId !== null && fork.currentTurnId !== event.turnId) {
        logger.error(`[WorkingState] STALE TURN_COMPLETED: got turnId=${event.turnId} but currentTurnId=${fork.currentTurnId} on fork ${event.forkId ?? 'root'}`)
        return fork
      }

      let turnWantsContinue: boolean

      if (event.result.success) {
        // Use turn decision from agent definition's turn policy
        turnWantsContinue = event.result.turnDecision === 'continue'
      } else {
        turnWantsContinue = !event.result.cancelled
      }

      // 'finish' means agent is done — ignore queued messages, don't continue
      const isFinished = event.result.success && event.result.turnDecision === 'finish'

      // If soft-interrupted, override willContinue to false
      let willContinue = isFinished ? false : (fork.softInterrupted ? false : (fork.hasQueuedMessages || fork.pendingWake || fork.pendingSeeVerdict || turnWantsContinue))

      const newFork: ForkWorkingState = {
        ...fork,
        working: false,
        willContinue,
        hasQueuedMessages: false,
        pendingWake: false,
        currentTurnAllowsDirectUserReply: false,
        currentTurnId: null,
        currentChainId: willContinue ? fork.currentChainId : null,
        pendingSeeVerdict: false,
      }

      // Emit stability first so downstream projections (e.g. AgentRegistry) update
      // status to idle before shouldTriggerChanged wakes the parent fork
      if (isStable(newFork) && !isStable(fork)) {
        emit.forkBecameStable({ forkId: event.forkId, timestamp: event.timestamp })
      }

      if (shouldTrigger(newFork) !== shouldTrigger(fork)) {
        emit.shouldTriggerChanged({
          forkId: event.forkId,
          shouldTrigger: shouldTrigger(newFork),
          chainId: newFork.currentChainId
        })
      }

      // Emit soft interrupt resolved if this fork was soft-interrupted and is now stable
      if (fork.softInterrupted && isStable(newFork) && event.forkId !== null) {
        emit.softInterruptResolved({ forkId: event.forkId })
      }

      return newFork
    },

    // Unexpected error during turn - clear all flags, go stable
    turn_unexpected_error: ({ event, fork, emit }) => {
      if (fork.currentTurnId !== null && fork.currentTurnId !== event.turnId) {
        logger.error(`[WorkingState] STALE TURN_UNEXPECTED_ERROR: got turnId=${event.turnId} but currentTurnId=${fork.currentTurnId} on fork ${event.forkId ?? 'root'}`)
        return fork
      }

      const newFork: ForkWorkingState = {
        ...fork,
        working: false,
        willContinue: false,
        hasQueuedMessages: false,
        currentTurnAllowsDirectUserReply: false,
        currentTurnId: null,
        currentChainId: null,
      }

      if (shouldTrigger(newFork) !== shouldTrigger(fork)) {
        emit.shouldTriggerChanged({
          forkId: event.forkId,
          shouldTrigger: shouldTrigger(newFork),
          chainId: newFork.currentChainId
        })
      }

      if (isStable(newFork) && !isStable(fork)) {
        emit.forkBecameStable({ forkId: event.forkId, timestamp: event.timestamp })
      }

      return newFork
    },

    interrupt: ({ event, fork, emit }) => {
      const interruptedTurnId = fork.currentTurnId
      const interruptedChainId = fork.currentChainId

      const newFork: ForkWorkingState = {
        ...fork,
        working: false,
        willContinue: false,
        hasQueuedMessages: false,
        pendingWake: false,
        currentTurnAllowsDirectUserReply: false,
        currentTurnId: null,
        currentChainId: null,
        compactionPending: false,
        contextLimitBlocked: false,
        softInterrupted: false,
      }

      if (isStable(newFork) && !isStable(fork)) {
        emit.forkBecameStable({ forkId: event.forkId, timestamp: event.timestamp })
      }

      if (shouldTrigger(newFork) !== shouldTrigger(fork)) {
        emit.shouldTriggerChanged({
          forkId: event.forkId,
          shouldTrigger: shouldTrigger(newFork),
          chainId: newFork.currentChainId
        })
      }

      return newFork
    },

    // Soft interrupt: prevent new turns but let current turn finish
    soft_interrupt: ({ event, fork, emit }) => {
      const newFork: ForkWorkingState = {
        ...fork,
        willContinue: false,
        softInterrupted: true
      }

      if (shouldTrigger(newFork) !== shouldTrigger(fork)) {
        emit.shouldTriggerChanged({
          forkId: event.forkId,
          shouldTrigger: shouldTrigger(newFork),
          chainId: newFork.currentChainId
        })
      }

      return newFork
    },

    agent_created: ({ event, fork, emit }) => {
      const newFork: ForkWorkingState = {
        ...fork,
        parentForkId: event.parentForkId,
        willContinue: true
      }

      emit.shouldTriggerChanged({
        forkId: event.forkId,
        shouldTrigger: true,
        chainId: null
      })

      return newFork
    },

  },

  globalEventHandlers: {
    turn_unexpected_error: ({ event, state, emit }) => {
      const subFork = state.forks.get(event.forkId)
      if (!subFork) return state
      const parentId = subFork.parentForkId

      const parentFork = state.forks.get(parentId)
      if (!parentFork) return state

      const newParentFork: ForkWorkingState = {
        ...parentFork,
        willContinue: parentFork.working ? parentFork.willContinue : true,
        pendingWake: true,
      }

      if (shouldTrigger(newParentFork) && !shouldTrigger(parentFork)) {
        emit.shouldTriggerChanged({
          forkId: parentId,
          shouldTrigger: true,
          chainId: newParentFork.currentChainId,
        })
      }

      return {
        ...state,
        forks: new Map(state.forks).set(parentId, newParentFork),
      }
    },

    turn_completed: ({ event, state, emit }) => {
      if (event.forkId === null) return state

      const subFork = state.forks.get(event.forkId)
      if (!subFork) return state
      if (!isStable(subFork)) return state

      const parentId = subFork.parentForkId
      const parentFork = state.forks.get(parentId)
      if (!parentFork) return state

      const newParentFork: ForkWorkingState = {
        ...parentFork,
        willContinue: true,
        hasQueuedMessages: parentFork.hasQueuedMessages || parentFork.working,
      }

      if (shouldTrigger(newParentFork) && !shouldTrigger(parentFork)) {
        emit.shouldTriggerChanged({
          forkId: parentId,
          shouldTrigger: true,
          chainId: newParentFork.currentChainId,
        })
      }

      return {
        ...state,
        forks: new Map(state.forks).set(parentId, newParentFork),
      }
    },

    subagent_user_killed: ({ event, state, emit }) => {
      const parentId = event.parentForkId
      if (!parentId) return state

      const parentFork = state.forks.get(parentId)
      if (!parentFork) return state

      const newParentFork: ForkWorkingState = {
        ...parentFork,
        willContinue: true,
      }

      if (shouldTrigger(newParentFork) && !shouldTrigger(parentFork)) {
        emit.shouldTriggerChanged({
          forkId: parentId,
          shouldTrigger: true,
          chainId: newParentFork.currentChainId,
        })
      }

      return {
        ...state,
        forks: new Map(state.forks).set(parentId, newParentFork),
      }
    },
  },

  signalHandlers: (on) => [
    // Gate turns when context limit is hit (proactive or reactive)
    on(CompactionProjection.signals.contextLimitBlockedChanged, ({ value, state, emit }) => {
      const { forkId, blocked } = value

      const forkState = state.forks.get(forkId)
      if (!forkState) return state

      const oldShouldTrigger = shouldTrigger(forkState)

      const newForkState: ForkWorkingState = {
        ...forkState,
        contextLimitBlocked: blocked
      }

      const newShouldTrigger = shouldTrigger(newForkState)

      if (newShouldTrigger !== oldShouldTrigger) {
        emit.shouldTriggerChanged({
          forkId,
          shouldTrigger: newShouldTrigger,
          chainId: newForkState.currentChainId
        })
      }

      // Emit stable if became stable
      if (isStable(newForkState) && !isStable(forkState)) {
        emit.forkBecameStable({ forkId, timestamp: value.timestamp })
      }

      return {
        ...state,
        forks: new Map(state.forks).set(forkId, newForkState)
      }
    }),

    // Gate turns when compaction is ready to finalize
    on(CompactionProjection.signals.compactionPendingChanged, ({ value, state, emit }) => {
      const { forkId, pending } = value

      const forkState = state.forks.get(forkId)
      if (!forkState) return state

      const oldShouldTrigger = shouldTrigger(forkState)

      const newForkState: ForkWorkingState = {
        ...forkState,
        compactionPending: pending
      }

      const newShouldTrigger = shouldTrigger(newForkState)

      if (newShouldTrigger !== oldShouldTrigger) {
        emit.shouldTriggerChanged({
          forkId,
          shouldTrigger: newShouldTrigger,
          chainId: newForkState.currentChainId
        })
      }

      if (isStable(newForkState) && !isStable(forkState)) {
        emit.forkBecameStable({ forkId, timestamp: value.timestamp })
      }

      return {
        ...state,
        forks: new Map(state.forks).set(forkId, newForkState)
      }
    }),


    on(UserMessageResolutionProjection.signals.userMessageResolved, ({ value, state, emit }) => {
      const forkState = state.forks.get(value.forkId)
      if (!forkState) return state

      const isDirectToSubagent = value.forkId !== null
      const contentText = value.content.map(part => part.type === 'text' ? part.text : '').join('')

      const newForkState: ForkWorkingState = {
        ...forkState,
        willContinue: true,
        hasQueuedMessages: forkState.hasQueuedMessages || forkState.working,
        pendingInboundCommunications: isDirectToSubagent
          ? [
              ...forkState.pendingInboundCommunications,
              {
                id: createId(),
                source: 'user',
                replyPolicy: 'user_reply_once',
                direction: 'from_agent',
                agentId: 'user',
                forkId: value.forkId,
                content: contentText,
                preview: toPreview(contentText),
                timestamp: value.timestamp,
                arrivedAtTurnId: forkState.currentTurnId,
              }
            ]
          : forkState.pendingInboundCommunications,
      }

      if (shouldTrigger(newForkState) !== shouldTrigger(forkState)) {
        emit.shouldTriggerChanged({
          forkId: value.forkId,
          shouldTrigger: shouldTrigger(newForkState),
          chainId: newForkState.currentChainId
        })
      }

      return {
        ...state,
        forks: new Map(state.forks).set(value.forkId, newForkState)
      }
    }),

    // Wake orchestrator (root) when sub-agent responds
    on(AgentRoutingProjection.signals.agentResponse, ({ value, state, emit }) => {
      const forkId = value.targetForkId  // null = root (orchestrator)
      const forkState = state.forks.get(forkId)
      if (!forkState) return state

      const newForkState: ForkWorkingState = {
        ...forkState,
        willContinue: true,
        hasQueuedMessages: forkState.hasQueuedMessages || forkState.working,
        pendingInboundCommunications: [
          ...forkState.pendingInboundCommunications,
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
            arrivedAtTurnId: forkState.currentTurnId,
          }
        ]
      }

      if (shouldTrigger(newForkState) && !shouldTrigger(forkState)) {
        emit.shouldTriggerChanged({
          forkId,
          shouldTrigger: true,
          chainId: newForkState.currentChainId
        })
      }

      return {
        ...state,
        forks: new Map(state.forks).set(forkId, newForkState)
      }
    }),

    // Wake sub-agent when orchestrator sends it a message
    on(AgentRoutingProjection.signals.agentMessage, ({ value, state, emit }) => {
      const forkId = value.targetForkId
      const forkState = state.forks.get(forkId)
      if (!forkState) return state

      const newForkState: ForkWorkingState = {
        ...forkState,
        willContinue: true,
        hasQueuedMessages: forkState.hasQueuedMessages || forkState.working,
        pendingInboundCommunications: [
          ...forkState.pendingInboundCommunications,
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
            arrivedAtTurnId: forkState.currentTurnId,
          }
        ]
      }

      if (shouldTrigger(newForkState) && !shouldTrigger(forkState)) {
        emit.shouldTriggerChanged({
          forkId,
          shouldTrigger: true,
          chainId: newForkState.currentChainId
        })
      }

      return {
        ...state,
        forks: new Map(state.forks).set(forkId, newForkState)
      }
    }),


  ]
})
