import { describe, expect, test } from 'bun:test'
import { buildRestoredQueuedInputValue, nextBulkInsertEpochForPaste } from './chat-controller'

describe('bulk-insert epoch helpers', () => {
  test('bumps epoch only when requested by paste effects', () => {
    expect(nextBulkInsertEpochForPaste(3, true)).toBe(4)
    expect(nextBulkInsertEpochForPaste(3, false)).toBe(3)
  })
})

describe('restoredQueuedInputText state', () => {
  test('builds same cleared composer shape with restored text and cursor at end', () => {
    const value = buildRestoredQueuedInputValue('queued message')
    expect(value.text).toBe('queued message')
    expect(value.cursorPosition).toBe('queued message'.length)
    expect(value.pasteSegments).toEqual([])
    expect(value.mentionSegments).toEqual([])
    expect(value.selectedPasteSegmentId).toBeNull()
    expect(value.selectedMentionSegmentId).toBeNull()
  })
})
