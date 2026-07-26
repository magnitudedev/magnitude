import { describe, expect, test } from 'vitest'
import { contextUsageWidth, formatContextUsage } from './context-usage'

describe('composer context usage', () => {
  test('shows current and total usage followed by the derived percentage', () => {
    expect(formatContextUsage(11_000, 220_000)).toBe('11k / 220k (5%)')
  })

  test('rounds both token values to whole thousands with halves rounded up', () => {
    expect(formatContextUsage(15_499, 200_499)).toBe('15k / 200k (8%)')
    expect(formatContextUsage(15_500, 200_500)).toBe('16k / 201k (8%)')
    expect(formatContextUsage(15_800, 200_000)).toBe('16k / 200k (8%)')
  })

  test('keeps current usage when the model limit is unavailable', () => {
    expect(formatContextUsage(11_000, null)).toBe('11k')
  })

  test('keeps the known context limit when usage has not arrived', () => {
    expect(formatContextUsage(null, 220_000)).toBe('— / 220k')
  })

  test('shows a compact fallback when neither usage nor limit has arrived', () => {
    expect(formatContextUsage(null, null)).toBe('—')
  })

  test('reports stable layout width including compacting arrows', () => {
    expect(contextUsageWidth(11_000, 220_000, false)).toBe('11k / 220k (5%)'.length)
    expect(contextUsageWidth(11_000, 220_000, true)).toBe('11k / 220k (5%)'.length + 8)
  })
})
