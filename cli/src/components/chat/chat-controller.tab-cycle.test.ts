import { describe, expect, test } from 'bun:test'

import { getCycledTabSelection, shouldCycleSubagentTabs } from './chat-controller'

describe('shouldCycleSubagentTabs', () => {
  test('allows plain Tab and Shift+Tab when menus are closed', () => {
    expect(shouldCycleSubagentTabs({
      keyName: 'tab',
      ctrl: false,
      meta: false,
      option: false,
      fileMentionOpen: false,
      slashMenuOpen: false,
    })).toBe(true)
  })

  test('blocks Tab cycling when mention menu is open', () => {
    expect(shouldCycleSubagentTabs({
      keyName: 'tab',
      ctrl: false,
      meta: false,
      option: false,
      fileMentionOpen: true,
      slashMenuOpen: false,
    })).toBe(false)
  })

  test('blocks Tab cycling when slash menu is open', () => {
    expect(shouldCycleSubagentTabs({
      keyName: 'tab',
      ctrl: false,
      meta: false,
      option: false,
      fileMentionOpen: false,
      slashMenuOpen: true,
    })).toBe(false)
  })
})

describe('getCycledTabSelection', () => {
  const tabs = [{ forkId: 'fork-1' }, { forkId: 'fork-2' }, { forkId: 'fork-3' }]

  test('Tab from Main selects first subagent', () => {
    expect(getCycledTabSelection({ selectedForkId: null, subagentTabs: tabs, shift: false })).toBe('fork-1')
  })

  test('Tab advances in subagent tab order', () => {
    expect(getCycledTabSelection({ selectedForkId: 'fork-1', subagentTabs: tabs, shift: false })).toBe('fork-2')
    expect(getCycledTabSelection({ selectedForkId: 'fork-2', subagentTabs: tabs, shift: false })).toBe('fork-3')
  })

  test('Tab from last subagent wraps to Main', () => {
    expect(getCycledTabSelection({ selectedForkId: 'fork-3', subagentTabs: tabs, shift: false })).toBeNull()
  })

  test('Shift+Tab from Main wraps to last subagent', () => {
    expect(getCycledTabSelection({ selectedForkId: null, subagentTabs: tabs, shift: true })).toBe('fork-3')
  })

  test('missing selected id falls back to Main index then cycles', () => {
    expect(getCycledTabSelection({ selectedForkId: 'stale-fork', subagentTabs: tabs, shift: false })).toBe('fork-1')
    expect(getCycledTabSelection({ selectedForkId: 'stale-fork', subagentTabs: tabs, shift: true })).toBe('fork-3')
  })

  test('single-tab case (Main only) remains on Main', () => {
    expect(getCycledTabSelection({ selectedForkId: null, subagentTabs: [], shift: false })).toBeNull()
    expect(getCycledTabSelection({ selectedForkId: null, subagentTabs: [], shift: true })).toBeNull()
  })
})
