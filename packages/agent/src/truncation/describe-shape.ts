/**
 * Budget-aware structural shape summary.
 *
 * Produces a JSON-like representation that shows real values when they fit
 * the budget and <angle-bracket descriptors> when they don't.
 */

import { CHARS_PER_TOKEN_UPPER } from '../constants'
import { charsToTokensUpper, allocateBudget } from './budget'
import { estimateText } from './estimate'
import type { Measurement } from './budget'
import { measureBounded } from './json/measure'
import type { JsonValue } from '@magnitudedev/ai'
import { formatSize } from './format'

const DEFAULT_BUDGET = 500

const INDENT = '  '

/**
 * Produce a budget-aware structural summary of a value.
 *
 * Small values render as real JSON. Large values get `<type, size>` descriptors.
 * Objects show quoted keys, arrays show item counts, strings show char/token counts.
 */
export function describeShape(value: unknown, budgetTokens: number = DEFAULT_BUDGET): string {
  return renderValue(value as JsonValue, budgetTokens, 0)
}

function renderValue(value: JsonValue, budget: number, depth: number): string {
  if (value === null) return 'null'
  if (value === undefined) return 'undefined'
  if (typeof value === 'boolean') return String(value)
  if (typeof value === 'number') return String(value)
  if (typeof value === 'string') return renderString(value, budget)
  if (Array.isArray(value)) return renderArray(value, budget, depth)
  if (typeof value === 'object') return renderObject(value as Record<string, JsonValue>, budget, depth)
  return '...'
}

function renderString(s: string, budget: number): string {
  const full = JSON.stringify(s)
  const fullCost = charsToTokensUpper(full.length)

  if (fullCost <= budget) return full

  const budgetChars = budget * CHARS_PER_TOKEN_UPPER

  if (s.length > budgetChars * 4) {
    // Way too big — descriptor with token estimate
    const tokens = formatSize(estimateText(s))
    return `<string, ${s.length} chars, ~${tokens} tokens>`
  }

  // Moderately over — show truncated prefix
  const availableChars = budgetChars - 5 // quote + ... + quote
  if (availableChars <= 0) return `<string, ${s.length} chars>`
  const prefix = s.slice(0, Math.floor(availableChars))
  return JSON.stringify(prefix).slice(0, -1) + '..."'
}

function renderArray(arr: JsonValue[], budget: number, depth: number): string {
  if (arr.length === 0) return '[]'

  // Check if entire array fits
  const measured = measureBounded(arr, budget)
  if (!measured.exceeded) return renderFull(arr, depth)

  // Framing cost: "[<N items>\n" + indent + "]\n" + "...N more\n"
  const framingCost = 3
  const available = budget - framingCost
  if (available <= 0) return `[<${arr.length} items>...]`

  const isHomogeneous = checkHomogeneous(arr)

  // Decide item count
  let itemCount: number
  let minPerItem: number
  if (isHomogeneous) {
    itemCount = Math.min(2, arr.length)
    minPerItem = 20
  } else {
    itemCount = Math.min(5, arr.length)
    minPerItem = 10
  }

  while (itemCount > 1 && (available / itemCount) < minPerItem) {
    itemCount--
  }

  if (available < minPerItem) return `[<${arr.length} items>...]`

  // Allocate budget across items
  const separatorCost = (itemCount - 1)
  const itemBudget = available - separatorCost
  const measurements = arr.slice(0, itemCount).map(v => measureBounded(v, itemBudget))
  const allocations = allocateBudget(measurements, itemBudget)

  const ind = INDENT.repeat(depth + 1)
  const baseInd = INDENT.repeat(depth)

  let result = `[<${arr.length} items>\n`
  for (let i = 0; i < itemCount; i++) {
    const rendered = renderValue(arr[i], allocations[i], depth + 1)
    const comma = (i < itemCount - 1 || itemCount < arr.length) ? ',' : ''
    result += ind + indentSubsequentLines(rendered, depth + 1) + comma + '\n'
  }
  if (itemCount < arr.length) {
    result += ind + `...${arr.length - itemCount} more\n`
  }
  result += baseInd + ']'

  return result
}

function renderObject(obj: Record<string, JsonValue>, budget: number, depth: number): string {
  const entries = Object.entries(obj).filter(([, v]) => v !== undefined)
  if (entries.length === 0) return '{}'

  // Check if entire object fits
  const measured = measureBounded(obj, budget)
  if (!measured.exceeded) return renderFull(obj, depth)

  const framingCost = 2 // braces
  const available = budget - framingCost
  if (available <= 0) return `{<${entries.length} keys>...}`

  // Calculate per-key overhead
  const keyOverheads = entries.map(([k], i) => {
    const sep = i > 0 ? 1 : 0 // comma
    return charsToTokensUpper(k.length + 4) + sep // "key": 
  })

  // Phase 1: find how many entries we can show with meaningful values
  let entriesToShow = entries.length
  while (entriesToShow > 0) {
    const structCost = keyOverheads.slice(0, entriesToShow).reduce((a, b) => a + b, 0)
    const valueMins = entries.slice(0, entriesToShow).map(([, v]) => minValueCost(v))
    const remainderCost = entriesToShow < entries.length ? 2 : 0
    if (structCost + valueMins.reduce((a, b) => a + b, 0) + remainderCost <= available) break
    entriesToShow--
  }

  if (entriesToShow === 0) return `{<${entries.length} keys>...}`

  // Phase 2: distribute remaining budget to values
  const totalKeyOverhead = keyOverheads.slice(0, entriesToShow).reduce((a, b) => a + b, 0)
  const remainderCost = entriesToShow < entries.length ? 2 : 0
  const valueBudget = available - totalKeyOverhead - remainderCost

  const measurements = entries.slice(0, entriesToShow).map(([, v]) => measureBounded(v, valueBudget))
  const allocations = allocateBudget(measurements, valueBudget)

  const ind = INDENT.repeat(depth + 1)
  const baseInd = INDENT.repeat(depth)

  let result = '{\n'
  for (let i = 0; i < entriesToShow; i++) {
    const [key, value] = entries[i]
    const rendered = renderValue(value, allocations[i], depth + 1)
    const comma = (i < entriesToShow - 1 || entriesToShow < entries.length) ? ',' : ''
    result += ind + `"${key}": ${indentSubsequentLines(rendered, depth + 1)}${comma}\n`
  }
  if (entriesToShow < entries.length) {
    result += ind + `...${entries.length - entriesToShow} more\n`
  }
  result += baseInd + '}'

  return result
}

/**
 * Minimum cost for a value to be rendered meaningfully.
 */
function minValueCost(value: JsonValue): number {
  if (value === null || value === undefined) return 1
  if (typeof value === 'boolean') return 1
  if (typeof value === 'number') return 1
  if (typeof value === 'string') {
    const fullCost = charsToTokensUpper(value.length + 2)
    return fullCost <= 3 ? fullCost : 3 // at minimum a descriptor or truncated prefix
  }
  if (Array.isArray(value)) {
    return value.length === 0 ? 1 : 2 // "[]" or "[<N items>...]"
  }
  const keys = Object.keys(value as Record<string, JsonValue>)
  return keys.length === 0 ? 1 : 2 // "{}" or "{<N keys>...}"
}

/**
 * Render a value as proper indented JSON (used when it fits the budget).
 */
function renderFull(value: JsonValue, depth: number): string {
  if (value === null) return 'null'
  if (value === undefined) return 'undefined'
  if (typeof value === 'boolean') return String(value)
  if (typeof value === 'number') return String(value)
  if (typeof value === 'string') return JSON.stringify(value)

  if (Array.isArray(value)) {
    if (value.length === 0) return '[]'
    const ind = INDENT.repeat(depth + 1)
    const baseInd = INDENT.repeat(depth)
    const items = value.map((v, i) => {
      const comma = i < value.length - 1 ? ',' : ''
      return ind + indentSubsequentLines(renderFull(v, depth + 1), depth + 1) + comma
    })
    return '[\n' + items.join('\n') + '\n' + baseInd + ']'
  }

  if (typeof value === 'object') {
    const entries = Object.entries(value as Record<string, JsonValue>).filter(([, v]) => v !== undefined)
    if (entries.length === 0) return '{}'
    const ind = INDENT.repeat(depth + 1)
    const baseInd = INDENT.repeat(depth)
    const items = entries.map(([k, v], i) => {
      const comma = i < entries.length - 1 ? ',' : ''
      return ind + `"${k}": ${indentSubsequentLines(renderFull(v, depth + 1), depth + 1)}${comma}`
    })
    return '{\n' + items.join('\n') + '\n' + baseInd + '}'
  }

  return String(value)
}

/**
 * Check if an array is homogeneous (all items are objects with the same keys).
 */
function checkHomogeneous(arr: JsonValue[]): boolean {
  if (arr.length < 2) return false
  const sample = arr.slice(0, Math.min(5, arr.length))
  if (!sample.every(item => typeof item === 'object' && item !== null && !Array.isArray(item))) {
    return false
  }
  const firstKeys = Object.keys(sample[0] as object).sort().join(',')
  return sample.every(item => Object.keys(item as object).sort().join(',') === firstKeys)
}

/**
 * Indent all lines of a multi-line string except the first.
 */
function indentSubsequentLines(text: string, depth: number): string {
  const lines = text.split('\n')
  if (lines.length <= 1) return text
  const ind = INDENT.repeat(depth)
  return lines[0] + '\n' + lines.slice(1).map(l => ind + l).join('\n')
}
