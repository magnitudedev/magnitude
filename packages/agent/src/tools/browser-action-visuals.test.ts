import { describe, expect, test } from 'bun:test'
import { formatBrowserActionVisual, getBrowserActionIcon } from './browser-action-visuals'

describe('browser action visuals', () => {
  test('maps known browser icons by toolKey', () => {
    expect(getBrowserActionIcon('navigate')).toBe('→')
    expect(getBrowserActionIcon('doubleClick')).toBe('◎◎')
    expect(getBrowserActionIcon('evaluate')).toBe('▶')
  })

  test('formats action-specific details', () => {
    expect(formatBrowserActionVisual('type', { content: 'hello world' })).toEqual({
      icon: '⌨',
      label: 'Type',
      detail: '"hello world"',
    })
    expect(formatBrowserActionVisual('switchTab', { index: 2 })).toEqual({
      icon: '⇥',
      label: 'Switch tab',
      detail: '#2',
    })
  })

  test('falls back safely for unknown browser key', () => {
    expect(formatBrowserActionVisual('futureBrowserAction', {})).toEqual({
      icon: '◎',
      label: 'Browser action',
    })
  })
})
