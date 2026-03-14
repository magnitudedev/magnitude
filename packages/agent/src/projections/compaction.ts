/**
 * CompactionProjection (Forked)
 *
 * Owns compaction-related state per fork: token estimates, compaction lifecycle flags,
 * and context limit blocking. Extracted from MemoryProjection to break circular dependencies.
 *
 * Key behavior:
 * - Tracks token estimates and whether compaction should be triggered
 * - Gates turns via contextLimitBlocked when compaction is in progress at hard cap
 * - Stores pending compaction data between compaction_ready and compaction_completed
 */

import { Projection, Signal } from '@magnitudedev/event-core'
import { peekSlot } from '@magnitudedev/providers'
import type { AppEvent, SessionContext } from '../events'
import { AgentRoutingProjection } from './agent-routing'

import { getContextLimits } from '../constants'
import { CHARS_PER_TOKEN } from '../constants'
import { SYSTEM_PROMPT_TOKENS } from '../generated/system-prompt-size'
import { buildSessionContextContent } from '../prompts'
import { ContentPart, textOf } from '../content'

// =============================================================================
// Context Limit Helpers
// =============================================================================

/** Compute whether turns should be blocked due to context limit */
function computeContextLimitBlocked(isCompacting: boolean, pendingFinalization: boolean, tokenEstimate: number): boolean {
  return (isCompacting || pendingFinalization) && tokenEstimate >= getContextLimits().hardCap
}

/** Estimate tokens for content string or content parts */
function estimateContentTokens(content: string): number
function estimateContentTokens(content: ContentPart[]): number
function estimateContentTokens(content: string | ContentPart[]): number {
  if (typeof content === 'string') {
    return Math.ceil(content.length / CHARS_PER_TOKEN)
  }
  let tokens = 0
  for (const part of content) {
    switch (part.type) {
      case 'text':
        tokens += Math.ceil(part.text.length / CHARS_PER_TOKEN)
        break
      case 'image':
        tokens += getImageTokenEstimator()(part.width, part.height)
        break
    }
  }
  return tokens
}

type ImageTokenEstimator = (width: number, height: number) => number

const estimators: Record<string, ImageTokenEstimator> = {
  // Anthropic: (w * h) / 750, with 1568px long-edge resize
  anthropic: (w, h) => {
    const longEdge = Math.max(w, h)
    if (longEdge > 1568) {
      const scale = 1568 / longEdge
      w = Math.round(w * scale)
      h = Math.round(h * scale)
    }
    return Math.ceil((w * h) / 750)
  },

  // OpenAI: tile-based (resize to 2048 max, 768 short side, 512x512 tiles)
  // Formula: 85 base + 170 per tile
  openai: (w, h) => {
    // Step 1: scale to fit within 2048x2048
    const maxDim = Math.max(w, h)
    if (maxDim > 2048) {
      const scale = 2048 / maxDim
      w = Math.round(w * scale)
      h = Math.round(h * scale)
    }
    // Step 2: if BOTH dimensions > 768, scale so shortest side = 768
    const minDim = Math.min(w, h)
    if (minDim > 768) {
      const scale = 768 / minDim
      w = Math.round(w * scale)
      h = Math.round(h * scale)
    }
    // Step 3: count 512x512 tiles
    const tilesW = Math.ceil(w / 512)
    const tilesH = Math.ceil(h / 512)
    return (tilesW * tilesH * 170) + 85
  },

  // Google: fixed per image (~258-560 tokens depending on model, use 560 as safe upper bound)
  google: () => 560,
  'google-ai': () => 560,
  'vertex-ai': () => 560,
}

function getImageTokenEstimator(): ImageTokenEstimator {
  const model = peekSlot('primary')?.model
  const modelId = model?.id ?? ''
  const providerId = model?.providerId ?? 'anthropic'

  // 1. Model name heuristic (most reliable)
  if (/claude/i.test(modelId)) return estimators.anthropic
  if (/gpt|o[1-9]-/i.test(modelId)) return estimators.openai
  if (/gemini/i.test(modelId)) return estimators.google

  // 2. Fall back to provider ID for unknown model names
  if (providerId === 'anthropic' || providerId === 'aws-bedrock') return estimators.anthropic
  if (providerId === 'openai') return estimators.openai
  if (providerId === 'google' || providerId === 'google-ai' || providerId === 'vertex-ai') return estimators.google

  // 3. Default to Anthropic (safe upper bound)
  return estimators.anthropic
}

// =============================================================================
// Types
// =============================================================================

/** Per-fork compaction state */
export interface ForkCompactionState {
  readonly tokenEstimate: number
  readonly shouldCompact: boolean
  readonly isCompacting: boolean
  readonly pendingFinalization: boolean
  readonly contextLimitBlocked: boolean
  readonly pendingCompactionData: {
    readonly summary: string
    readonly compactedMessageCount: number
    readonly originalTokenEstimate: number
    readonly refreshedContext: SessionContext | null
  } | null
}

// =============================================================================
// Projection
// =============================================================================

export const CompactionProjection = Projection.defineForked<AppEvent, ForkCompactionState>()({
  name: 'Compaction',

  reads: [AgentRoutingProjection] as const,

  signals: {
    shouldCompactChanged: Signal.create<{ forkId: string | null; shouldCompact: boolean }>('Compaction/shouldCompactChanged'),
    compactionPendingChanged: Signal.create<{ forkId: string | null; pending: boolean }>('Compaction/compactionPendingChanged'),
    contextLimitBlockedChanged: Signal.create<{ forkId: string | null; blocked: boolean }>('Compaction/contextLimitBlockedChanged'),
  },

  initialFork: {
    tokenEstimate: 0,
    shouldCompact: false,
    isCompacting: false,
    pendingFinalization: false,
    contextLimitBlocked: false,
    pendingCompactionData: null,
  },

  eventHandlers: {
    session_initialized: ({ event, fork }) => {
      const content = buildSessionContextContent(event.context)
      const contentTokens = estimateContentTokens(content)
      const tokenEstimate = SYSTEM_PROMPT_TOKENS + contentTokens
      return {
        ...fork,
        tokenEstimate,
        shouldCompact: tokenEstimate > getContextLimits().softCap,
      }
    },

    user_message: ({ event, fork, emit }) => {
      const addedTokens = estimateContentTokens(event.content)
      const newTokenEstimate = fork.tokenEstimate + addedTokens
      const newShouldCompact = newTokenEstimate > getContextLimits().softCap
      if (newShouldCompact !== fork.shouldCompact) {
        emit.shouldCompactChanged({ forkId: event.forkId, shouldCompact: newShouldCompact })
      }
      const newBlocked = computeContextLimitBlocked(fork.isCompacting, fork.pendingFinalization, newTokenEstimate)
      if (newBlocked !== fork.contextLimitBlocked) {
        emit.contextLimitBlockedChanged({ forkId: event.forkId, blocked: newBlocked })
      }
      return {
        ...fork,
        tokenEstimate: newTokenEstimate,
        shouldCompact: newShouldCompact,
        contextLimitBlocked: newBlocked,
      }
    },

    turn_completed: ({ event, fork, emit }) => {
      let addedTokens = 0
      // Estimate tokens from response parts
      for (const part of event.responseParts) {
        if (part.type === 'text' || part.type === 'thinking') {
          addedTokens += estimateContentTokens(part.content)
        }
      }
      for (const tc of event.toolCalls) {
        if (tc.result.status === 'success') {
          const output = typeof tc.result.output === 'string' ? tc.result.output : (JSON.stringify(tc.result.output) ?? '')
          addedTokens += estimateContentTokens(output)
        }
      }

      const newTokenEstimate = event.inputTokens !== null
        ? event.inputTokens + addedTokens
        : fork.tokenEstimate + addedTokens
      const newShouldCompact = newTokenEstimate > getContextLimits().softCap
      if (newShouldCompact !== fork.shouldCompact) {
        emit.shouldCompactChanged({ forkId: event.forkId, shouldCompact: newShouldCompact })
      }
      const newBlocked = computeContextLimitBlocked(fork.isCompacting, fork.pendingFinalization, newTokenEstimate)
      if (newBlocked !== fork.contextLimitBlocked) {
        emit.contextLimitBlockedChanged({ forkId: event.forkId, blocked: newBlocked })
      }
      return {
        ...fork,
        tokenEstimate: newTokenEstimate,
        shouldCompact: newShouldCompact,
        contextLimitBlocked: newBlocked,
      }
    },

    turn_unexpected_error: ({ event, fork }) => {
      const addedTokens = estimateContentTokens(event.message)
      return {
        ...fork,
        tokenEstimate: fork.tokenEstimate + addedTokens,
      }
    },

    compaction_started: ({ event, fork, emit }) => {
      const newBlocked = computeContextLimitBlocked(true, fork.pendingFinalization, fork.tokenEstimate)
      if (newBlocked !== fork.contextLimitBlocked) {
        emit.contextLimitBlockedChanged({ forkId: event.forkId, blocked: newBlocked })
      }
      return {
        ...fork,
        isCompacting: true,
        contextLimitBlocked: newBlocked,
      }
    },

    compaction_ready: ({ event, fork, emit }) => {
      emit.compactionPendingChanged({ forkId: event.forkId, pending: true })
      return {
        ...fork,
        pendingFinalization: true,
        pendingCompactionData: {
          summary: event.summary,
          compactedMessageCount: event.compactedMessageCount,
          originalTokenEstimate: event.originalTokenEstimate,
          refreshedContext: event.refreshedContext,
        },
      }
    },

    compaction_completed: ({ event, fork, emit }) => {
      emit.compactionPendingChanged({ forkId: event.forkId, pending: false })
      if (fork.contextLimitBlocked) {
        emit.contextLimitBlockedChanged({ forkId: event.forkId, blocked: false })
      }

      const tokenEstimate = Math.max(0, fork.tokenEstimate - event.tokensSaved)
      const shouldCompact = tokenEstimate > getContextLimits().softCap

      if (shouldCompact) {
        emit.shouldCompactChanged({ forkId: event.forkId, shouldCompact: true })
      }

      return {
        ...fork,
        tokenEstimate,
        shouldCompact,
        isCompacting: false,
        pendingFinalization: false,
        contextLimitBlocked: false,
        pendingCompactionData: null,
      }
    },

    compaction_failed: ({ event, fork, emit }) => {
      if (fork.contextLimitBlocked) {
        emit.contextLimitBlockedChanged({ forkId: event.forkId, blocked: false })
      }
      return {
        ...fork,
        isCompacting: false,
        pendingFinalization: false,
        contextLimitBlocked: false,
        pendingCompactionData: null,
      }
    },

    context_limit_hit: ({ event, fork, emit }) => {
      if (!fork.contextLimitBlocked) {
        emit.contextLimitBlockedChanged({ forkId: event.forkId, blocked: true })
      }
      if (!fork.isCompacting && !fork.pendingFinalization) {
        emit.shouldCompactChanged({ forkId: event.forkId, shouldCompact: true })
      }
      return {
        ...fork,
        contextLimitBlocked: true,
      }
    },
  },

  signalHandlers: (on) => [
    on(AgentRoutingProjection.signals.agentRegistered, ({ value, state }) => {
      const { forkId, parentForkId } = value
      const parentState = state.forks.get(parentForkId)
      if (!parentState) {
        throw new Error(`Parent fork ${parentForkId} not found in CompactionProjection`)
      }

      const newForkState: ForkCompactionState = {
        tokenEstimate: parentState.tokenEstimate,
        shouldCompact: false,
        isCompacting: false,
        pendingFinalization: false,
        contextLimitBlocked: false,
        pendingCompactionData: null,
      }

      return {
        ...state,
        forks: new Map(state.forks).set(forkId, newForkState),
      }
    }),
  ],
})