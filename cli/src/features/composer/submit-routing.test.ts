import { describe, expect, test } from 'bun:test'
import { shouldHandleSlashCommandInTab } from '@magnitudedev/client-common'

describe('shouldHandleSlashCommand in task/main-chat context', () => {
  test('allows slash commands in main chat', () => {
    expect(shouldHandleSlashCommandInTab(null)).toBe(true)
  })

  test('keeps slash commands main-chat only in task context', () => {
    expect(shouldHandleSlashCommandInTab('fork-123')).toBe(false)
  })
})
