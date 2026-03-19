import { describe, expect, test } from 'bun:test'
import { buildSubmitDispatchEvents, shouldHandleSlashCommandInTab } from './submit-routing'

describe('buildSubmitDispatchEvents', () => {
  test('root submit targets root only', () => {
    expect(buildSubmitDispatchEvents(null)).toEqual([
      { type: 'user_message', forkId: null },
    ])
  })

  test('subagent submit interrupts first, then sends targeted user message', () => {
    expect(buildSubmitDispatchEvents('fork-123')).toEqual([
      { type: 'interrupt', forkId: 'fork-123' },
      { type: 'user_message', forkId: 'fork-123' },
    ])
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