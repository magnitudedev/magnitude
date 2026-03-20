import { describe, expect, test } from 'bun:test'
import { getEffectiveSelectedForkId } from './app'

describe('getEffectiveSelectedForkId', () => {
  test('returns null when selected fork is removed from subagent tabs', () => {
    expect(
      getEffectiveSelectedForkId('fork-removed', [
        { forkId: 'fork-a' },
        { forkId: 'fork-b' },
      ])
    ).toBeNull()
  })

  test('preserves selection when selected fork is still present', () => {
    expect(
      getEffectiveSelectedForkId('fork-b', [
        { forkId: 'fork-a' },
        { forkId: 'fork-b' },
      ])
    ).toBe('fork-b')
  })
})