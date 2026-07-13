import { CHARS_PER_TOKEN_UPPER, CHARS_PER_TOKEN_LOWER } from '../constants'

/** Result of measuring a value's size against a cap */
export type Measurement = {
  size: number  // in tokens
  exceeded: boolean
}

/** Convert chars to tokens using upper bound (conservative — for truncation fitting) */
export function charsToTokensUpper(chars: number): number {
  return Math.ceil(chars / CHARS_PER_TOKEN_UPPER)
}

/** Convert chars to tokens using lower bound (safer higher estimate — for context accounting) */
export function charsToTokensLower(chars: number): number {
  return Math.ceil(chars / CHARS_PER_TOKEN_LOWER)
}

/**
 * Distribute budget (in tokens) fairly across items using smallest-first allocation.
 *
 * Algorithm:
 * 1. Sort items by size (smallest first)
 * 2. For each item in sorted order:
 *    - Calculate equal share: remaining / count
 *    - If item fits in share: give exact size needed, save remainder
 *    - If item exceeds share: give equal share
 * 3. Return allocations in original order (in tokens)
 */
export function allocateBudget(measurements: Measurement[], budgetTokens: number): number[] {
  const n = measurements.length
  if (n === 0) return []

  const allocations = new Array(n).fill(0)

  // Create index array and sort by size (smallest first)
  const indices = measurements.map((_, i) => i)
  indices.sort((a, b) => measurements[a].size - measurements[b].size)

  let remaining = budgetTokens
  let count = n

  for (const i of indices) {
    const share = remaining / count
    const m = measurements[i]

    if (!m.exceeded && m.size <= share) {
      allocations[i] = m.size
      remaining -= m.size
    } else {
      allocations[i] = Math.floor(share)
      remaining -= Math.floor(share)
    }
    count--
  }

  return allocations
}
