import { describe, expect, it } from 'vitest'
import type { DisplayRootStatus } from '@magnitudedev/sdk'
import {
  formatTokenCount,
  rootDetailSegments,
} from './root-detail'

type DisplayRootDetail = Extract<DisplayRootStatus, { readonly _tag: 'Working' }>['detail']

const prefill = (
  overrides: Partial<Extract<DisplayRootDetail, { readonly _tag: 'Prefill' }>> = {},
): Extract<DisplayRootDetail, { readonly _tag: 'Prefill' }> => ({
  _tag: 'Prefill',
  completedTokens: 0,
  totalTokens: 0,
  cachedTokens: 0,
  ...overrides,
})

describe('root detail copy', () => {
  it('formats token counts compactly', () => {
    expect(formatTokenCount(820)).toBe('820')
    expect(formatTokenCount(1_100)).toBe('1.1k')
    expect(formatTokenCount(14_300)).toBe('14.3k')
  })

  it('describes a cold prefill using total progress', () => {
    expect(rootDetailSegments(prefill({
      completedTokens: 9_400,
      totalTokens: 14_300,
    }))).toEqual({
      keyword: 'Prefill',
      detail: '9.4k / 14.3k tokens',
      trailing: null,
    })
  })

  it('separates cached tokens from the input tokens being prefilled', () => {
    expect(rootDetailSegments(prefill({
      completedTokens: 14_020,
      totalTokens: 14_300,
      cachedTokens: 13_200,
    }))).toEqual({
      keyword: 'Prefill',
      detail: '820 / 1.1k tokens',
      trailing: '13.2k cached',
    })
  })

  it('structures cached progress for a single activity rail line', () => {
    expect(rootDetailSegments(prefill({
      completedTokens: 14_020,
      totalTokens: 14_300,
      cachedTokens: 13_200,
    }))).toEqual({
      keyword: 'Prefill',
      detail: '820 / 1.1k tokens',
      trailing: '13.2k cached',
    })
  })
})
