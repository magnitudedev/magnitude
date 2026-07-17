import { describe, expect, it } from 'vitest'

import {
  formatModelUsage,
  formatReset,
  formatUsageSummary,
  usagePercentLeft,
} from './usage'

describe('usage presentation', () => {
  it('presents the percentage remaining', () => {
    expect(usagePercentLeft(0, 2_000)).toBe(100)
    expect(usagePercentLeft(20, 2_000)).toBe(99)
    expect(usagePercentLeft(500, 2_000)).toBe(75)
    expect(usagePercentLeft(2_100, 2_000)).toBe(0)
    expect(usagePercentLeft(100, 0)).toBe(0)
  })

  it('shows useful reset precision', () => {
    expect(formatReset(3 * 60 * 60 * 1000 + 55 * 60 * 1000)).toBe('3h 55m')
    expect(formatReset(2 * 24 * 60 * 60 * 1000 + 4 * 60 * 60 * 1000)).toBe('2d 4h')
  })

  it('formats period and model usage without dollar amounts', () => {
    const summary = formatUsageSummary({
      requestCount: 12,
      inputTokens: 1_250_000,
      outputTokens: 82_000,
    })
    const model = formatModelUsage({
      requestCount: 4,
      inputTokens: 420_000,
      outputTokens: 31_000,
    })

    expect(summary).toBe('12 requests  ·  1.3M input tokens  ·  82k output tokens')
    expect(model).toBe('4 requests  ·  420k in / 31k out')
    expect(`${summary}${model}`).not.toContain('$')
  })
})
