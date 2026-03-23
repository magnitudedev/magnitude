import { describe, expect, test } from 'bun:test'
import { getEffectiveSelectedForkId, getSelectedForkContentVersion } from './app'

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

describe('getSelectedForkContentVersion', () => {
  test('uses main sentinel when no subagent is selected', () => {
    expect(getSelectedForkContentVersion(null, null)).toBe('main')
  })

  test('includes selected fork id and content counts', () => {
    expect(
      getSelectedForkContentVersion('fork-a', {
        messages: [{ id: 'm1' } as any, { id: 'm2' } as any],
        pendingInboundCommunications: [{ id: 'p1' } as any],
      })
    ).toBe('fork-a:2:1')
  })

  test('falls back to zero counts before fork display is populated', () => {
    expect(getSelectedForkContentVersion('fork-a', null)).toBe('fork-a:0:0')
  })
})