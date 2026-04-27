/**
 * Usage Calculation — shared utilities for building CallUsage from raw token counts.
 *
 * Extracted from model-proxy.ts so both model-proxy and provider-client can use them.
 */

import type { ProviderModel } from './model/model'
import type { CallUsage } from './state/provider-state'
import { getModelCost } from './registry'

/**
 * Calculate costs from token counts using the model's pricing.
 * OAuth subscriptions are free ($0). Returns null costs if no pricing available.
 */
export function calculateCosts(
  model: ProviderModel | null,
  authType: string | null,
  inputTokens: number | null,
  outputTokens: number | null,
  cacheReadTokens: number | null,
  cacheWriteTokens: number | null,
): { inputCost: number | null; outputCost: number | null; totalCost: number | null } {
  // Subscription-based providers cost $0
  if (authType === 'oauth') {
    return { inputCost: 0, outputCost: 0, totalCost: 0 }
  }

  if (!model) return { inputCost: null, outputCost: null, totalCost: null }

  const pricing = getModelCost(model.providerId, model.id)
  if (!pricing) return { inputCost: null, outputCost: null, totalCost: null }

  let inputCost: number | null = null
  let outputCost: number | null = null

  if (inputTokens !== null) {
    inputCost = (inputTokens / 1_000_000) * pricing.inputPerM
    if (cacheReadTokens !== null && pricing.cacheReadPerM != null) {
      inputCost += (cacheReadTokens / 1_000_000) * pricing.cacheReadPerM
    }
    if (cacheWriteTokens !== null && pricing.cacheWritePerM != null) {
      inputCost += (cacheWriteTokens / 1_000_000) * pricing.cacheWritePerM
    }
  }

  if (outputTokens !== null) {
    outputCost = (outputTokens / 1_000_000) * pricing.outputPerM
  }

  const totalCost = (inputCost !== null || outputCost !== null)
    ? (inputCost ?? 0) + (outputCost ?? 0)
    : null

  return { inputCost, outputCost, totalCost }
}

/**
 * Build a complete CallUsage from raw token counts.
 * Calculates costs based on the model's pricing.
 */
export function buildUsage(
  model: ProviderModel | null,
  authType: string | null,
  inputTokens: number | null,
  outputTokens: number | null,
  cacheReadTokens: number | null,
  cacheWriteTokens: number | null,
): CallUsage {
  const costs = calculateCosts(model, authType, inputTokens, outputTokens, cacheReadTokens, cacheWriteTokens)
  return { inputTokens, outputTokens, cacheReadTokens, cacheWriteTokens, ...costs }
}