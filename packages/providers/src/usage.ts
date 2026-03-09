/**
 * Usage Calculation — shared utilities for building CallUsage from raw token counts.
 *
 * Extracted from model-proxy.ts so both model-proxy and provider-client can use them.
 */

import type { ResolvedModel, CallUsage } from './provider-state'
import { getModelCost } from './registry'

/**
 * Calculate costs from token counts using the model's pricing.
 * OAuth subscriptions are free ($0). Returns null costs if no pricing available.
 */
export function calculateCosts(
  resolved: ResolvedModel | null,
  inputTokens: number | null,
  outputTokens: number | null,
  cacheReadTokens: number | null,
  cacheWriteTokens: number | null,
): { inputCost: number | null; outputCost: number | null; totalCost: number | null } {
  // Subscription-based providers cost $0
  if (resolved?.auth?.type === 'oauth') {
    return { inputCost: 0, outputCost: 0, totalCost: 0 }
  }

  if (!resolved) return { inputCost: null, outputCost: null, totalCost: null }

  const pricing = getModelCost(resolved.providerId, resolved.modelId)
  if (!pricing) return { inputCost: null, outputCost: null, totalCost: null }

  let inputCost: number | null = null
  let outputCost: number | null = null

  if (inputTokens !== null) {
    inputCost = (inputTokens / 1_000_000) * pricing.input
    if (cacheReadTokens !== null && pricing.cache_read) {
      inputCost += (cacheReadTokens / 1_000_000) * pricing.cache_read
    }
    if (cacheWriteTokens !== null && pricing.cache_write) {
      inputCost += (cacheWriteTokens / 1_000_000) * pricing.cache_write
    }
  }

  if (outputTokens !== null) {
    outputCost = (outputTokens / 1_000_000) * pricing.output
  }

  const totalCost = (inputCost !== null || outputCost !== null)
    ? (inputCost ?? 0) + (outputCost ?? 0)
    : null

  return { inputCost, outputCost, totalCost }
}

/**
 * Build a complete CallUsage from raw token counts.
 * Calculates costs based on the resolved model's pricing.
 */
export function buildUsage(
  resolved: ResolvedModel | null,
  inputTokens: number | null,
  outputTokens: number | null,
  cacheReadTokens: number | null,
  cacheWriteTokens: number | null,
): CallUsage {
  const costs = calculateCosts(resolved, inputTokens, outputTokens, cacheReadTokens, cacheWriteTokens)
  return { inputTokens, outputTokens, cacheReadTokens, cacheWriteTokens, ...costs }
}
