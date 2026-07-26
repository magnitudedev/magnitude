import { describe, expect, test } from 'vitest'
import { contextUsageWidth, formatContextUsage } from './context-usage'

describe('composer context usage', () => {
  test('shows current and total usage followed by the derived percentage', () => {
    expect(formatContextUsage(11_000, 220_000)).toBe('11k / 220k ctx (5%)')
  })

  test('keeps current usage when the model limit is unavailable', () => {
    expect(formatContextUsage(11_000, null)).toBe('11k ctx')
  })

  test('keeps the known context limit when usage has not arrived', () => {
    expect(formatContextUsage(null, 220_000)).toBe('— / 220k ctx')
  })

  test('shows a compact fallback when neither usage nor limit has arrived', () => {
    expect(formatContextUsage(null, null)).toBe('— ctx')
  })

  test('reports stable layout width including compacting arrows', () => {
    expect(contextUsageWidth(11_000, 220_000, false)).toBe('11k / 220k ctx (5%)'.length)
    expect(contextUsageWidth(11_000, 220_000, true)).toBe('11k / 220k ctx (5%)'.length + 8)
  })
})
