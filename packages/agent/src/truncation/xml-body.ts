import { CHARS_PER_TOKEN_UPPER } from '../constants'
import { allocateBudget, charsToTokensUpper } from './budget'
import { measureBounded } from './json/measure'
import { truncate } from './json/truncate'
import type { JsonValue } from '@magnitudedev/ai'

export function truncateXmlBodyString(value: string, budgetTokens: number): string {
  if (charsToTokensUpper(value.length) <= budgetTokens) return value
  if (budgetTokens < 1) return '...'

  const maxOutputChars = Math.floor(budgetTokens * CHARS_PER_TOKEN_UPPER)
  const availableForContent = maxOutputChars - 3
  if (availableForContent <= 0) return '...'
  return `${value.slice(0, availableForContent)}...`
}

export function renderXmlBodyValue(value: JsonValue, budgetTokens: number): string {
  if (typeof value === 'string') return truncateXmlBodyString(value, budgetTokens)
  return truncate(value, budgetTokens)
}

export function renderXmlBodyValues(values: readonly JsonValue[], totalBudgetTokens: number): string[] {
  const measurements = values.map(value => measureBounded(value, totalBudgetTokens))
  const allocations = allocateBudget(measurements, totalBudgetTokens)
  return values.map((value, index) => renderXmlBodyValue(value, allocations[index] ?? 0))
}
