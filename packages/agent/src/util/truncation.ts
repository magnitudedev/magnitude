/**
 * Budget-Aware JSON Truncation
 *
 * Truncates JSON values to fit within a token budget while preserving structure
 * and distributing budget fairly across items.
 *
 * Key concepts:
 * - Budget is in approximate tokens (1 token ≈ 3 chars)
 * - Structure is guaranteed first (key names, collection counts), then remaining budget fills content
 * - Small items shown fully, large items share remaining budget
 * - Remainder indicators show how much was truncated ("...N more")
 */

// Local type definition (matches common package)
export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue }

import { CHARS_PER_TOKEN } from '../constants'

// Token costs for common structural elements
const TOKEN_COSTS = {
  ELLIPSIS: 1,           // "..."
  ARRAY_BRACKETS: 1,     // "[]"
  OBJECT_BRACES: 1,      // "{}"
  SEPARATOR: 1,          // ", "
  COLON: 1,              // ": "
  ARRAY_PLACEHOLDER: 2,  // "[...]"
  OBJECT_PLACEHOLDER: 2, // "{...}"
  REMAINDER_MAX: 7,      // ", ...9999 more" (worst case)
} as const

/** Convert chars to tokens */
export function charsToTokens(chars: number): number {
  return Math.ceil(chars / CHARS_PER_TOKEN)
}

/** Result of measuring a value */
export type Measurement = {
  size: number  // in tokens
  exceeded: boolean
}

/**
 * Count the JSON-escaped length of a single character by char code.
 * Conservative: overcounts surrogate pairs (counts 6 each instead of 1),
 * but NEVER undercounts. This ensures truncation output never exceeds budget.
 */
function jsonEscapedCharLen(code: number): number {
  if (code === 0x22 || code === 0x5C) return 2
  if (code < 0x20) {
    if (code === 0x08 || code === 0x09 || code === 0x0A || code === 0x0C || code === 0x0D) return 2
    return 6
  }
  if (code >= 0xD800 && code <= 0xDFFF) return 6
  return 1
}

/**
 * Measure the serialized size of a value in tokens, stopping early if it exceeds the cap.
 *
 * Why bounded? If budget is 500 tokens and value is 50k tokens, we only need to know
 * "too big" - not exactly how big. This makes measurement O(min(size, cap)).
 */
export function measureBounded(value: JsonValue, capTokens: number): Measurement {
  const capChars = capTokens * CHARS_PER_TOKEN
  let count = 0

  function measure(v: unknown): boolean {
    if (count > capChars) return false

    if (v === null) {
      count += 4 // "null"
      return true
    }
    if (v === undefined) {
      count += 9 // "undefined"
      return true
    }
    if (typeof v === 'boolean') {
      count += v ? 4 : 5 // "true" or "false"
      return true
    }
    if (typeof v === 'number') {
      count += String(v).length
      return true
    }
    if (typeof v === 'string') {
      count += 2 // quotes
      for (let j = 0; j < v.length; j++) {
        count += jsonEscapedCharLen(v.charCodeAt(j))
        if (count > capChars) return false
      }
      return count <= capChars
    }

    if (Array.isArray(v)) {
      count += 2 // []
      for (let i = 0; i < v.length; i++) {
        if (i > 0) count += 2 // ", "
        if (!measure(v[i])) return false
      }
      return count <= capChars
    }

    if (typeof v === 'object') {
      count += 2 // {}
      const entries = Object.entries(v)
      for (let i = 0; i < entries.length; i++) {
        if (i > 0) count += 2 // ", "
        count += entries[i][0].length + 2 // key + ": "
        if (!measure(entries[i][1])) return false
      }
      return count <= capChars
    }

    return true
  }

  const completed = measure(value)
  return {
    size: charsToTokens(Math.min(count, capChars)),
    exceeded: !completed || count > capChars
  }
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
      // Fits in share - give exact amount needed
      allocations[i] = m.size
      remaining -= m.size
    } else {
      // Too big - give equal share
      allocations[i] = Math.floor(share)
      remaining -= Math.floor(share)
    }
    count--
  }

  return allocations
}

/**
 * Leaf-level cost: the cheapest USEFUL representation of a value.
 * O(1), non-recursive. Used as the value cost within one-level lookahead.
 */
function flatMinCost(value: JsonValue): number {
  if (value === null || value === undefined) return charsToTokens(4)
  if (typeof value === 'boolean') return charsToTokens(value ? 4 : 5)
  if (typeof value === 'number') return charsToTokens(String(value).length)
  if (typeof value === 'string') {
    const fullCost = charsToTokens(value.length + 2)
    if (fullCost <= 3) return fullCost
    return 3
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return TOKEN_COSTS.ARRAY_BRACKETS
    return charsToTokens(`[...${value.length} items]`.length)
  }
  const keys = Object.keys(value as Record<string, JsonValue>)
  if (keys.length === 0) return TOKEN_COSTS.OBJECT_BRACES
  return charsToTokens(`{...(${keys.length} keys)}`.length)
}

/**
 * Calculate the minimum budget for a USEFUL representation of a value.
 * Used in Phase 1 of structure-first truncation to determine how many
 * entries/items can be shown.
 *
 * For non-object types: O(1) flat cost (same as flatMinCost).
 * For non-empty objects: one-level lookahead — estimates the cost of showing
 * min(3, numKeys) entries with flatMinCost values. This is O(min(3, k)),
 * NOT recursive, and gives much more accurate estimates for nested objects.
 *
 * Why one level? The old recursive approach walked the entire tree (O(tree_size))
 * and gave 20 tokens per leaf → explosive costs → showed nothing.
 * Flat O(1) underestimates nested objects (~5 tokens) → Phase 1 packs too many
 * entries → values get bare `...`. One level is the principled middle ground:
 * accurate enough to prevent value starvation, bounded enough to never explode.
 */
function minMeaningfulCost(value: JsonValue): number {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return flatMinCost(value)
  }

  const entries = Object.entries(value as Record<string, JsonValue>)
  if (entries.length === 0) return TOKEN_COSTS.OBJECT_BRACES

  const previewCount = Math.min(3, entries.length)
  let cost = TOKEN_COSTS.OBJECT_BRACES

  for (let i = 0; i < previewCount; i++) {
    const [key, val] = entries[i]
    if (i > 0) cost += TOKEN_COSTS.SEPARATOR
    cost += charsToTokens(key.length) + TOKEN_COSTS.COLON
    cost += flatMinCost(val)
  }

  if (previewCount < entries.length) {
    cost += TOKEN_COSTS.REMAINDER_MAX
  }

  return cost
}

/**
 * Render a JSON value fully without any truncation.
 * Produces the same format as truncate() does for values that fit within budget.
 * Only call after measureBounded confirms the value fits.
 */
function renderFull(value: JsonValue): string {
  if (value === undefined) return 'undefined'
  if (value === null) return 'null'
  if (typeof value === 'boolean') return String(value)
  if (typeof value === 'number') return String(value)
  if (typeof value === 'string') return JSON.stringify(value)
  if (Array.isArray(value)) {
    return '[' + value.map(v => renderFull(v)).join(', ') + ']'
  }
  const entries = Object.entries(value as Record<string, JsonValue>)
    .filter(([, v]) => v !== undefined)
  return '{' + entries.map(([k, v]) => `${k}: ${renderFull(v)}`).join(', ') + '}'
}

/**
 * Truncate a value to fit within a token budget.
 *
 * Returns a string representation that:
 * - Fits within the budget
 * - Preserves structure (valid JSON-ish)
 * - Shows remainder indicators for truncated collections
 */
export function truncate(value: JsonValue, budgetTokens: number): string {
  // Clamp budget to type minimum — structured types need at least their placeholder cost
  if (Array.isArray(value) && value.length > 0) {
    budgetTokens = Math.max(budgetTokens, TOKEN_COSTS.ARRAY_PLACEHOLDER)
  } else if (typeof value === 'object' && value !== null && Object.keys(value).length > 0) {
    budgetTokens = Math.max(budgetTokens, TOKEN_COSTS.OBJECT_PLACEHOLDER)
  }

  if (budgetTokens < TOKEN_COSTS.ELLIPSIS) return '...'

  // Primitives
  if (value === undefined) return budgetTokens >= 2 ? 'undefined' : '...'
  if (value === null) return budgetTokens >= 2 ? 'null' : '...'

  if (typeof value === 'boolean') {
    const s = String(value)
    return charsToTokens(s.length) <= budgetTokens ? s : '...'
  }

  if (typeof value === 'number') {
    const s = String(value)
    return charsToTokens(s.length) <= budgetTokens ? s : '...'
  }

  if (typeof value === 'string') {
    return truncateString(value, budgetTokens)
  }

  if (Array.isArray(value)) {
    return truncateArray(value, budgetTokens)
  }

  if (typeof value === 'object') {
    return truncateObject(value as Record<string, JsonValue>, budgetTokens)
  }

  // Should be unreachable for valid JsonValue
  return '...'
}

function truncateString(s: string, budgetTokens: number): string {
  const full = JSON.stringify(s)
  if (charsToTokens(full.length) <= budgetTokens) return full

  if (budgetTokens < 3) return '...'

  // Output format: "content..." → 1 (open quote) + escapedLen + 4 (suffix ...")
  const maxOutputChars = Math.floor(budgetTokens * CHARS_PER_TOKEN)
  const availableForContent = maxOutputChars - 5

  if (availableForContent <= 0) return '...'

  let escapedLen = 0
  let rawEnd = 0
  for (let i = 0; i < s.length; i++) {
    const charLen = jsonEscapedCharLen(s.charCodeAt(i))
    if (escapedLen + charLen > availableForContent) break
    escapedLen += charLen
    rawEnd = i + 1
  }

  if (rawEnd === 0) return '...'

  const truncated = s.slice(0, rawEnd)
  return JSON.stringify(truncated).slice(0, -1) + '..."'
}

function truncateArray(arr: JsonValue[], budgetTokens: number): string {
  if (arr.length === 0) return '[]'

  // Phase 0: Real full-fit check using measureBounded
  if (!measureBounded(arr, budgetTokens).exceeded) {
    return renderFull(arr)
  }

  // Can we at least show "[...N items]"?
  const countIndicator = `[...${arr.length} items]`
  if (budgetTokens < TOKEN_COSTS.ARRAY_PLACEHOLDER) return '[...]'

  const availableTokens = budgetTokens - TOKEN_COSTS.ARRAY_BRACKETS - TOKEN_COSTS.REMAINDER_MAX
  if (availableTokens < TOKEN_COSTS.ELLIPSIS) {
    return charsToTokens(countIndicator.length) <= budgetTokens ? countIndicator : '[...]'
  }

  const isHomogeneous = checkHomogeneous(arr)

  if (isHomogeneous) {
    return truncateHomogeneousArray(arr, availableTokens)
  } else {
    return truncateHeterogeneousArray(arr, availableTokens)
  }
}

/**
 * Check if array items are homogeneous (same shape).
 * Samples first few items and compares their keys.
 */
function checkHomogeneous(arr: JsonValue[]): boolean {
  // Need at least 2 items to compare
  if (arr.length < 2) return false

  // Only check objects - primitives/mixed types are heterogeneous
  const sample = arr.slice(0, Math.min(5, arr.length))
  if (!sample.every(item => typeof item === 'object' && item !== null && !Array.isArray(item))) {
    return false
  }

  // Compare keys of first item to others
  const firstKeys = Object.keys(sample[0] as object).sort().join(',')
  return sample.every(item => Object.keys(item as object).sort().join(',') === firstKeys)
}

/**
 * Truncate homogeneous array: show fewer items with more detail.
 *
 * Since all items have the same shape, showing one or two fully is more
 * informative than showing many with no content.
 */
function truncateHomogeneousArray(arr: JsonValue[], availableTokens: number): string {
  const MIN_TOKENS_PER_ITEM = 15
  const itemCount = Math.min(3, arr.length, Math.max(1, Math.floor(availableTokens / MIN_TOKENS_PER_ITEM)))

  const separatorTokens = (itemCount - 1) * TOKEN_COSTS.SEPARATOR
  const tokensForItems = availableTokens - separatorTokens

  const measurements = arr.slice(0, itemCount).map(v => measureBounded(v, tokensForItems))
  const allocations = allocateBudget(measurements, tokensForItems)

  const parts: string[] = []
  for (let i = 0; i < itemCount; i++) {
    parts.push(truncate(arr[i], allocations[i]))
  }

  let result = '[' + parts.join(', ')
  if (itemCount < arr.length) {
    result += `, ...${arr.length - itemCount} more`
  }
  result += ']'

  return result
}

/**
 * Truncate heterogeneous array: show more items with less detail each.
 *
 * Since items have different shapes, we want to show more of them
 * to capture the variety, even if each item is heavily truncated.
 */
function truncateHeterogeneousArray(arr: JsonValue[], availableTokens: number): string {
  const sampleSize = Math.min(5, arr.length)
  let totalMinMeaningful = 0
  for (let i = 0; i < sampleSize; i++) {
    totalMinMeaningful += minMeaningfulCost(arr[i])
  }
  const avgMinMeaningful = totalMinMeaningful / sampleSize
  const targetPerItem = Math.max(15, Math.ceil(avgMinMeaningful * 3))

  const itemCount = Math.max(1, Math.min(arr.length, Math.floor(availableTokens / targetPerItem)))

  const separatorTokens = (itemCount - 1) * TOKEN_COSTS.SEPARATOR
  const tokensForItems = availableTokens - separatorTokens

  const measurements = arr.slice(0, itemCount).map(v => measureBounded(v, tokensForItems))
  const allocations = allocateBudget(measurements, tokensForItems)

  const parts: string[] = []
  for (let i = 0; i < itemCount; i++) {
    parts.push(truncate(arr[i], allocations[i]))
  }

  let result = '[' + parts.join(', ')
  if (itemCount < arr.length) {
    result += `, ...${arr.length - itemCount} more`
  }
  result += ']'

  return result
}

/**
 * Truncate an object using structure-first budgeted approach.
 *
 * Phase 0: Check if full object fits within budget.
 * Phase 1: Guarantee structure — fit as many key names as possible with placeholder values.
 * Phase 2: Distribute remaining budget to values for actual content.
 */
function truncateObject(obj: Record<string, JsonValue>, budgetTokens: number): string {
  const entries = Object.entries(obj)
  if (entries.length === 0) return '{}'

  // Phase 0: Real full-fit check
  if (!measureBounded(obj, budgetTokens).exceeded) {
    return renderFull(obj)
  }

  if (budgetTokens < TOKEN_COSTS.OBJECT_PLACEHOLDER) return '{...}'

  // Phase 1: STRUCTURE (guaranteed)
  const availableTokens = budgetTokens - TOKEN_COSTS.OBJECT_BRACES

  const keyOverheads = entries.map(([k], i) => {
    const sepTokens = i > 0 ? TOKEN_COSTS.SEPARATOR : 0
    return sepTokens + charsToTokens(k.length) + TOKEN_COSTS.COLON
  })

  const meaningfulCosts = entries.map(([, v]) => minMeaningfulCost(v))

  let entriesToShow = entries.length
  while (entriesToShow > 0) {
    const structuralCost = keyOverheads.slice(0, entriesToShow).reduce((a, b) => a + b, 0)
    const meaningfulTotal = meaningfulCosts.slice(0, entriesToShow).reduce((a, b) => a + b, 0)
    const remainderCost = entriesToShow < entries.length ? TOKEN_COSTS.REMAINDER_MAX : 0

    if (structuralCost + meaningfulTotal + remainderCost <= availableTokens) {
      break
    }
    entriesToShow--
  }

  if (entriesToShow === 0) {
    const countStr = `{...(${entries.length} keys)}`
    return charsToTokens(countStr.length) <= budgetTokens ? countStr : '{...}'
  }

  // Phase 2: CONTENT (best-effort)
  const totalKeyOverhead = keyOverheads.slice(0, entriesToShow).reduce((a, b) => a + b, 0)
  const remainderCost = entriesToShow < entries.length ? TOKEN_COSTS.REMAINDER_MAX : 0
  const valuesTokens = Math.max(0, availableTokens - totalKeyOverhead - remainderCost)

  const measurements = entries.slice(0, entriesToShow).map(([, v]) => measureBounded(v, valuesTokens))
  const allocations = allocateBudget(measurements, valuesTokens)

  let result = '{'
  let shown = 0

  for (let i = 0; i < entriesToShow; i++) {
    const [key, value] = entries[i]
    const allocation = allocations[i]

    const sep = shown > 0 ? ', ' : ''
    const valStr = truncate(value, allocation)

    result += `${sep}${key}: ${valStr}`
    shown++
  }

  if (shown < entries.length) {
    result += `, ...${entries.length - shown} more`
  }
  result += '}'

  return result
}

/**
 * Truncate multiple values, distributing budget fairly across them.
 *
 * Use this when you have multiple items that need to share a total budget.
 */
export function truncateMany(values: JsonValue[], totalBudgetTokens: number): string[] {
  if (values.length === 0) return []

  // Measure all values
  const measurements = values.map(v => measureBounded(v, totalBudgetTokens))

  // Allocate budget
  const allocations = allocateBudget(measurements, totalBudgetTokens)

  // Truncate each to its allocation
  return values.map((v, i) => truncate(v, allocations[i]))
}

/**
 * Format a size in tokens as a human-readable string.
 * e.g., 4000 -> "4k", 500 -> "500"
 */
export function formatSize(tokens: number): string {
  if (tokens < 1000) return String(tokens)
  const k = tokens / 1000
  return k >= 10 ? `${Math.round(k)}k` : `${k.toFixed(1)}k`
}
