/**
 * ContextUsageProjection (Forked)
 *
 * User-facing context display state (x/y), intentionally separate from compaction internals.
 */
import { Projection } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { CHARS_PER_TOKEN, getContextLimits } from '../constants'
import type { ContentPart } from '../content'

export type ContextUsageSource = 'provider' | 'mixed' | 'estimated'

export interface ContextUsageForkState {
  readonly retainedTokens: number
  readonly source: ContextUsageSource
  readonly hardCapTokens: number
  readonly lastProviderInputTokens: number | null
}

function estimateTextTokens(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN)
}

function estimateContentTokens(content: ContentPart[]): number {
  let tokens = 0
  for (const part of content) {
    if (part.type === 'text') tokens += estimateTextTokens(part.text)
  }
  return tokens
}

function estimateTurnFallbackIncrement(event: Extract<AppEvent, { type: 'turn_completed' }>): number {
  let addedTokens = 0

  for (const part of event.responseParts) {
    if (part.type === 'text' || part.type === 'thinking') addedTokens += estimateTextTokens(part.content)
  }

  for (const tc of event.toolCalls) {
    if (tc.result.status === 'error' || tc.result.status === 'rejected') {
      addedTokens += estimateTextTokens(tc.result.message)
    }
  }

  for (const observed of event.observedResults) {
    addedTokens += estimateContentTokens([...observed.content])
  }

  return addedTokens
}

export const ContextUsageProjection = Projection.defineForked<AppEvent, ContextUsageForkState>()({
  name: 'ContextUsage',

  initialFork: {
    retainedTokens: 0,
    source: 'estimated',
    hardCapTokens: getContextLimits().hardCap,
    lastProviderInputTokens: null,
  },

  eventHandlers: {
    session_initialized: ({ fork }) => ({
      ...fork,
      hardCapTokens: getContextLimits().hardCap,
    }),

    turn_completed: ({ event, fork }) => {
      if (event.inputTokens !== null) {
        const providerIncrement = fork.lastProviderInputTokens === null
          ? event.inputTokens
          : Math.max(0, event.inputTokens - fork.lastProviderInputTokens)

        return {
          ...fork,
          retainedTokens: Math.max(0, fork.retainedTokens + providerIncrement),
          source: 'provider',
          lastProviderInputTokens: event.inputTokens,
        }
      }

      const fallbackIncrement = estimateTurnFallbackIncrement(event)
      return {
        ...fork,
        retainedTokens: Math.max(0, fork.retainedTokens + fallbackIncrement),
        source: fork.lastProviderInputTokens === null ? 'estimated' : 'mixed',
      }
    },

    turn_unexpected_error: ({ event, fork }) => ({
      ...fork,
      retainedTokens: Math.max(0, fork.retainedTokens + estimateTextTokens(event.message)),
      source: fork.lastProviderInputTokens === null ? 'estimated' : 'mixed',
    }),

    compaction_completed: ({ event, fork }) => ({
      ...fork,
      retainedTokens: Math.max(0, fork.retainedTokens - event.tokensSaved),
    }),
  },
})
