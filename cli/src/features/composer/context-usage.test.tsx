import { describe, expect, test } from 'vitest'
import { contextUsageWidth, formatContextUsage } from './context-usage'

describe('composer context usage', () => {
  test('shows current usage followed by derived percentage without repeating the total', () => {
    expect(formatContextUsage(11_000, 220_000)).toBe('ctx 11k (5%)')
  })

  test('keeps current usage when the model limit is unavailable', () => {
    expect(formatContextUsage(11_000, null)).toBe('ctx 11k')
  })

  test('shows an explicit unknown value when usage has not arrived', () => {
    expect(formatContextUsage(null, 220_000)).toBe('ctx —')
  })

  test('reports stable layout width including compacting arrows', () => {
    expect(contextUsageWidth(11_000, 220_000, false)).toBe('ctx 11k (5%)'.length)
    expect(contextUsageWidth(11_000, 220_000, true)).toBe('ctx 11k (5%)'.length + 8)
  })
})
