import { describe, expect, test } from 'bun:test'
import {
  buildRestoredQueuedInputValue,
  nextBulkInsertEpochForPaste,
  shouldBumpBulkInsertEpoch,
} from './chat-controller'

describe('bulk-insert epoch helpers', () => {
  test('bumps epoch only for text-affecting paste outcomes', () => {
    expect(shouldBumpBulkInsertEpoch('text-inline')).toBe(true)
    expect(shouldBumpBulkInsertEpoch('text-segment')).toBe(true)
    expect(shouldBumpBulkInsertEpoch('clipboard-image')).toBe(false)
    expect(shouldBumpBulkInsertEpoch('pasted-image-path')).toBe(false)
    expect(shouldBumpBulkInsertEpoch('empty')).toBe(false)

    expect(nextBulkInsertEpochForPaste(3, 'text-inline')).toBe(4)
    expect(nextBulkInsertEpochForPaste(3, 'text-segment')).toBe(4)
    expect(nextBulkInsertEpochForPaste(3, 'clipboard-image')).toBe(3)
    expect(nextBulkInsertEpochForPaste(3, 'pasted-image-path')).toBe(3)
    expect(nextBulkInsertEpochForPaste(3, 'empty')).toBe(3)
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
