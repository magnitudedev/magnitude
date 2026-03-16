/**
 * WorkingStateProjection (Forked)
 *
 * Core state machine controlling the turn loop, per-fork.
 * Each fork has independent working/willContinue state.
 */

import { Signal, Projection } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { AgentRoutingProjection } from './agent-routing'
import { CompactionProjection } from './compaction'


// =============================================================================
// State
// =============================================================================

/** Per-fork working state */
export interface ForkWorkingState {
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
  readonly pendingMentionTimestamps: readonly number[]
}

// =============================================================================
// Derived
// =============================================================================

export const shouldTrigger = (state: ForkWorkingState): boolean =>
  !state.working && state.willContinue && !state.compactionPending && !state.contextLimitBlocked && state.pendingMentionTimestamps.length === 0

export const isStable = (state: ForkWorkingState): boolean =>
  !state.working && !state.willContinue && !state.compactionPending && !state.contextLimitBlocked

// =============================================================================
// Projection
// =============================================================================

export const WorkingStateProjection = Projection.defineForked<AppEvent, ForkWorkingState>()({
  name: 'WorkingState',

  initialFork: {
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
    pendingMentionTimestamps: []
  },

  signals: {
    shouldTriggerChanged: Signal.create<{ forkId: string | null; shouldTrigger: boolean; chainId: string | null }>('WorkingState/shouldTriggerChanged'),
    forkBecameStable: Signal.create<{ forkId: string | null; timestamp: number }>('WorkingState/forkBecameStable'),
    softInterruptResolved: Signal.create<{ forkId: string }>('WorkingState/softInterruptResolved'),
    turnInterrupted: Signal.create<{ forkId: string | null; turnId: string; chainId: string | null }>('WorkingState/turnInterrupted')
  },

  eventHandlers: {
    user_message: ({ event, fork, emit }) => {
      const isQueued = fork.working
      const hasMentions = (event.attachments ?? []).some(attachment => attachment.type === 'mention')

      const newFork: ForkWorkingState = {
        ...fork,
        willContinue: true,
        hasQueuedMessages: fork.hasQueuedMessages || isQueued,
        pendingMentionTimestamps: hasMentions
          ? [...fork.pendingMentionTimestamps, event.timestamp]
          : fork.pendingMentionTimestamps
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

    file_mention_resolved: ({ event, fork, emit }) => {
      const newFork: ForkWorkingState = {
        ...fork,
        pendingMentionTimestamps: fork.pendingMentionTimestamps.filter(
          timestamp => timestamp !== event.sourceMessageTimestamp
        )
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

    turn_started: ({ event, fork }) => ({
      ...fork,
      working: true,
      willContinue: false,
      pendingWake: false,
      hasQueuedMessages: false,
      currentTurnId: event.turnId,
      currentChainId: event.chainId
    }),

    turn_completed: ({ event, fork, emit }) => {
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
      let willContinue = isFinished ? false : (fork.softInterrupted ? false : (fork.hasQueuedMessages || fork.pendingWake || turnWantsContinue))

      const newFork: ForkWorkingState = {
        ...fork,
        working: false,
        willContinue,
        hasQueuedMessages: false,
        pendingWake: false,
        currentTurnId: null,
        currentChainId: willContinue ? fork.currentChainId : null
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
      const newFork: ForkWorkingState = {
        ...fork,
        working: false,
        willContinue: false,
        hasQueuedMessages: false,
        currentTurnId: null,
        currentChainId: null
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
        currentTurnId: null,
        currentChainId: null,
        compactionPending: false,
        contextLimitBlocked: false,
        softInterrupted: false
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

      if (interruptedTurnId !== null) {
        emit.turnInterrupted({
          forkId: event.forkId,
          turnId: interruptedTurnId,
          chainId: interruptedChainId
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
        willContinue: true
      }

      emit.shouldTriggerChanged({
        forkId: event.forkId,
        shouldTrigger: true,
        chainId: null
      })

      return newFork
    },


    // Stop completed fork from continuing
    agent_dismissed: ({ event, fork, emit }) => {
      const newFork: ForkWorkingState = {
        ...fork,
        willContinue: false,  // Fork is done, don't continue
        hasQueuedMessages: false,
        softInterrupted: false
      }

      if (shouldTrigger(newFork) !== shouldTrigger(fork)) {
        emit.shouldTriggerChanged({
          forkId: event.forkId,
          shouldTrigger: false,
          chainId: null
        })
      }

      // TODO: fork cleanup — need to design proper fork completion UX before removing state
      // return null
      return newFork
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
        emit.forkBecameStable({ forkId, timestamp: Date.now() })
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
        emit.forkBecameStable({ forkId, timestamp: Date.now() })
      }

      return {
        ...state,
        forks: new Map(state.forks).set(forkId, newForkState)
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
        hasQueuedMessages: forkState.hasQueuedMessages || forkState.working
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
        hasQueuedMessages: forkState.hasQueuedMessages || forkState.working
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
