import { describe, expect, test } from 'bun:test'
import { applyPasteIntent } from './apply'
import { derivePasteEffects } from './effects'
import type { InputValue } from '../../../types/store'

const EMPTY_INPUT: InputValue = {
  text: '',
  cursorPosition: 0,
  lastEditDueToNav: false,
  pasteSegments: [],
  mentionSegments: [],
  selectedPasteSegmentId: null,
  selectedMentionSegmentId: null,
}

describe('applyPasteIntent and derivePasteEffects', () => {
  test('inline insert mutates text and bumps bulk insert', () => {
    let input = EMPTY_INPUT
    const result = applyPasteIntent({
      intent: { kind: 'insert-inline-text', text: 'hello' },
      setInputValue: (updater) => {
        input = updater(input)
      },
    })

    expect(input.text).toBe('hello')
    const effects = derivePasteEffects(result)
    expect(effects.shouldBumpBulkInsertEpoch).toBe(true)
  })

  test('segment insert creates paste segment and bumps bulk insert', () => {
    let input = EMPTY_INPUT
    const result = applyPasteIntent({
      intent: { kind: 'insert-segment-text', text: 'very long text' },
      setInputValue: (updater) => {
        input = updater(input)
      },
    })

    expect(input.pasteSegments.length).toBe(1)
    const effects = derivePasteEffects(result)
    expect(effects.shouldBumpBulkInsertEpoch).toBe(true)
  })

  test('noop keeps bulk insert unchanged and emits feedback reason', () => {
    let input = EMPTY_INPUT
    const result = applyPasteIntent({
      intent: { kind: 'noop', reason: 'empty' },
      setInputValue: (updater) => {
        input = updater(input)
      },
    })

    expect(input.text).toBe('')
    const effects = derivePasteEffects(result)
    expect(effects.shouldBumpBulkInsertEpoch).toBe(false)
    expect(effects.feedbackReason).toBe('empty')
  })
})
