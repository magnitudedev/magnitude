import { describe, expect, it } from 'vitest'
import { formatElapsedMs, formatWorkDuration } from './format-elapsed'

describe('formatElapsedMs', () => {
  it('formats elapsed clocks as minutes and seconds', () => {
    expect(formatElapsedMs(65_900)).toBe('1:05')
  })
})

describe('formatWorkDuration', () => {
  it.each([
    [999, '<1 second'],
    [1_000, '1 second'],
    [5_999, '5 seconds'],
    [60_000, '1 minute'],
    [65_999, '1:05'],
    [120_000, '2 minutes'],
  ])('formats %i milliseconds as %s', (durationMs, expected) => {
    expect(formatWorkDuration(durationMs)).toBe(expected)
  })
})
