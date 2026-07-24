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
      primary: 'Loading conversation into the model · 9.4k / 14.3k tokens',
      secondary: null,
    })
  })

  it('separates reused context from newly added messages', () => {
    expect(modelRequestProgressCopy(prefill({
      completedTokens: 14_020,
      totalTokens: 14_300,
      cachedTokens: 13_200,
    }))).toEqual({
      primary: 'Adding new messages to the model · 820 / 1.1k tokens',
      secondary: 'Reusing 13.2k tokens from earlier messages',
    })
  })

  it('structures cached progress for a single activity rail line', () => {
    expect(modelRequestProgressSegments(prefill({
      completedTokens: 14_020,
      totalTokens: 14_300,
      cachedTokens: 13_200,
    }))).toEqual({
      label: 'Adding new messages',
      detail: '820 / 1.1k tokens',
      trailing: 'Reusing 13.2k tokens',
    })
  })
})
