/**
 * Compaction sizing helpers.
 *
 * Uses per-entry `estimatedTokens` from WindowEntry (computed at creation time).
 */

import type { WindowEntry } from '../window'
import { KEEP_MESSAGE_RATIO } from '../constants'

export function computeCompactionSizing(
  messages: readonly WindowEntry[],
  softCap: number,
): { compactedMessageCount: number; keptTailTokens: number } {
  const keepBudget = softCap * KEEP_MESSAGE_RATIO
  let accumulated = 0
  let keepCount = 0

  // Walk backwards from end, keeping messages until budget exceeded
  for (let i = messages.length - 1; i >= 1; i--) {
    const entryTokens = messages[i].estimatedTokens
    if (accumulated + entryTokens > keepBudget) break
    accumulated += entryTokens
    keepCount++
  }

  return {
    compactedMessageCount: Math.max(0, messages.length - 1 - keepCount),
    keptTailTokens: accumulated,
  }
}
