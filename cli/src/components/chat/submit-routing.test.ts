import { describe, expect, test } from 'bun:test'
import { buildSubmitDispatchEvents, shouldHandleSlashCommandInTab } from './submit-routing'

describe('buildSubmitDispatchEvents', () => {
  test('root submit targets root only', () => {
    expect(buildSubmitDispatchEvents(null)).toEqual([
      { type: 'user_message', forkId: null },
    ])
  })

  test('subagent submit sends targeted user message without interrupting', () => {
    const events = buildSubmitDispatchEvents('fork-123')
    expect(events).toEqual([
      { type: 'user_message', forkId: 'fork-123' },
    ])
    expect(events.some(event => event.type === 'interrupt')).toBe(false)
  })
})

describe('shouldHandleSlashCommandInTab', () => {
  test('allows slash commands in root tab', () => {
    expect(shouldHandleSlashCommandInTab(null)).toBe(true)
  })

  test('keeps slash commands root-only in subagent tabs', () => {
    expect(shouldHandleSlashCommandInTab('fork-123')).toBe(false)
  })
})