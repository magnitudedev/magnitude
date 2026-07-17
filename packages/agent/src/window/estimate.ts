import type { ContextPart } from '../content'
import type { CompletedTurn } from '../window/types'
import type { TimelineEntry } from './inbox/types'

import type { Skill } from '@magnitudedev/skills'
import type { RoleId } from '../agents/role-validation'

import { estimateContentTokens, estimateText } from '../truncation/estimate'
import { estimateCompletedTurn } from '../util/turn-estimation'
import { renderTimeline } from './inbox/render'
import { getAgentDefinition } from '../agents/registry'
import { buildSystemPrompt } from '../prompts/system-prompt-builder'

// =============================================================================
// Per-entry estimation
// =============================================================================

/** Estimate tokens for content-based entries (session_context, fork_context, compacted). */
export function estimateContentEntry(content: ContextPart[]): number {
  return estimateContentTokens(content)
}

/** Estimate tokens for an assistant turn entry. */
export function estimateTurnEntry(turn: CompletedTurn): number {
  return estimateCompletedTurn(turn)
}

/** Estimate tokens for a context (timeline) entry by rendering it the same way windowToPrompt does. */
export function estimateContextEntry(timeline: readonly TimelineEntry[]): number {
  if (timeline.length === 0) return 0
  const rendered = renderTimeline({ timeline, timezone: null })
  return estimateContentTokens(rendered)
}

/** Estimate tokens for an observer_turn entry. Uses a rough heuristic since we no longer render XML. */
export function estimateObserverTurnEntry(data: { readonly justification: string | null; readonly reasoning?: string | null }): number {
  // Rough estimate: justification + reasoning text + overhead for tool call structure
  return (data.justification ? estimateText(data.justification) : 0) + (data.reasoning ? estimateText(data.reasoning) : 0) + 80
}

/** Estimate tokens for an advisor_response entry. */
export function estimateAdvisorResponseEntry(data: { readonly content: string }): number {
  return estimateText(data.content) + 20
}

// =============================================================================
// System prompt estimation
// =============================================================================

const systemPromptTokenCache = new Map<string, number>()

/**
 * Estimate the token count of the full system prompt for a given role.
 * Cached per role+toolkit composition — system prompts are stable across turns.
 */
export function estimateSystemPromptTokens(
  roleId: RoleId,
  skills: Map<string, Skill>,
  options?: { solo?: boolean; systemPromptOverride?: string },
): number {
  const cacheKey = `${roleId}` + (options?.systemPromptOverride !== undefined ? '|override' : '')
  const cached = systemPromptTokenCache.get(cacheKey)
  if (cached !== undefined) return cached

  const agentDef = getAgentDefinition(roleId)
  const prompt = buildSystemPrompt({ roleDef: agentDef, skills, systemPromptOverride: options?.systemPromptOverride })

  const tokens = estimateText(prompt)
  systemPromptTokenCache.set(cacheKey, tokens)
  return tokens
}

// =============================================================================
// Budget math
// =============================================================================

/**
 * Compute total token estimate from anchor state + current messageTokens.
 *
 * If anchored: last API inputTokens measurement + delta since measurement.
 * If unanchored: systemPromptTokens + messageTokens (pure heuristic).
 */
export function computeTokenEstimate(
  systemPromptTokens: number,
  messageTokens: number,
  lastAnchoredTotal: number | null,
  lastAnchoredMessageTokens: number | null,
): number {
  if (lastAnchoredTotal !== null && lastAnchoredMessageTokens !== null) {
    return lastAnchoredTotal + (messageTokens - lastAnchoredMessageTokens)
  }
  return systemPromptTokens + messageTokens
}
