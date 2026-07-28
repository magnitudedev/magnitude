import { describe, expect, it } from 'vitest'
import type { DisplayModelRequestActivity } from '@magnitudedev/sdk'
import {
  formatTokenCount,
  modelRequestProgressCopy,
  modelRequestProgressSegments,
} from './model-request-progress'

const prefill = (
  overrides: Partial<DisplayModelRequestActivity> = {},
): DisplayModelRequestActivity => ({
  requestId: 'request-1',
  turnId: 'turn-1',
  forkId: null,
  startedAt: 0,
  phase: 'prefill',
  completedTokens: 0,
  totalTokens: 0,
  cachedTokens: 0,
  ...overrides,
})

describe('model request progress copy', () => {
  it('formats token counts compactly', () => {
    expect(formatTokenCount(820)).toBe('820')
    expect(formatTokenCount(1_100)).toBe('1.1k')
    expect(formatTokenCount(14_300)).toBe('14.3k')
  })

  it('describes a cold prefill using total progress', () => {
    expect(modelRequestProgressCopy(prefill({
      completedTokens: 9_400,
      totalTokens: 14_300,
    }))).toEqual({
      primary: 'Prefilling · 9.4k / 14.3k input tokens',
      secondary: null,
    })
  })

  it('separates cached tokens from the input tokens being prefilled', () => {
    expect(modelRequestProgressCopy(prefill({
      completedTokens: 14_020,
      totalTokens: 14_300,
      cachedTokens: 13_200,
    }))).toEqual({
      primary: 'Prefilling · 820 / 1.1k input tokens',
      secondary: 'Using 13.2k cached tokens',
    })
  })

  it('structures cached progress for a single activity rail line', () => {
    expect(modelRequestProgressSegments(prefill({
      completedTokens: 14_020,
      totalTokens: 14_300,
      cachedTokens: 13_200,
    }))).toEqual({
      label: 'Prefilling',
      detail: '820 / 1.1k input tokens',
      trailing: 'Using 13.2k cached tokens',
    })
  })
})
