import { describe, expect, test } from 'bun:test'
import { nextBulkInsertEpochForPaste } from './chat-controller'
import { resolvePasteIntent } from './paste/content-resolver'
import { applyPasteIntent } from './paste/apply'
import { derivePasteEffects } from './paste/effects'

describe('paste pipeline', () => {
  test('prefers event text and routes to inline insertion', async () => {
    const trace: string[] = []
    const intent = await resolvePasteIntent({
      eventText: 'pasted-text',
      readClipboardText: () => {
        trace.push('readClipboardText')
        return 'clipboard-text'
      },
      tryAddClipboardImage: async () => {
        trace.push('addClipboardImage')
        return true
      },
      tryAddImageFromFilePath: async () => {
        trace.push('addImageFromFilePath')
        return false
      },
      inlinePastePillCharLimit: 1000,
    })

    expect(intent).toEqual({ kind: 'insert-inline-text', text: 'pasted-text' })
    expect(trace).toEqual(['addImageFromFilePath'])
  })

  test('path image resolution wins before insertion', async () => {
    const intent = await resolvePasteIntent({
      eventText: '/tmp/image.png',
      readClipboardText: () => 'clipboard-text',
      tryAddClipboardImage: async () => false,
      tryAddImageFromFilePath: async () => true,
      inlinePastePillCharLimit: 1000,
    })
    expect(intent).toEqual({ kind: 'add-path-image', rawPath: '/tmp/image.png' })
  })

  test('empty text falls back to clipboard image and can noop', async () => {
    const intentNoImage = await resolvePasteIntent({
      eventText: '',
      readClipboardText: () => '',
      tryAddClipboardImage: async () => false,
      tryAddImageFromFilePath: async () => false,
      inlinePastePillCharLimit: 1000,
    })
    expect(intentNoImage).toEqual({ kind: 'noop', reason: 'empty' })

    const intentWithImage = await resolvePasteIntent({
      eventText: '',
      readClipboardText: () => '',
      tryAddClipboardImage: async () => true,
      tryAddImageFromFilePath: async () => false,
      inlinePastePillCharLimit: 1000,
    })
    expect(intentWithImage).toEqual({ kind: 'add-clipboard-image' })
  })

  test('apply/effects map inline vs segment + bulk insert epoch behavior', () => {
    let text = ''
    const inline = applyPasteIntent({
      intent: { kind: 'insert-inline-text', text: 'hello' },
      setInputValue: (updater) => {
        const next = updater({
          text,
          cursorPosition: text.length,
          lastEditDueToNav: false,
          pasteSegments: [],
          mentionSegments: [],
          selectedPasteSegmentId: null,
          selectedMentionSegmentId: null,
        })
        text = next.text
      },
    })
    expect(text).toBe('hello')
    expect(derivePasteEffects(inline).shouldBumpBulkInsertEpoch).toBe(true)

    const segment = applyPasteIntent({
      intent: { kind: 'insert-segment-text', text: 'this is long' },
      setInputValue: () => {},
    })
    expect(derivePasteEffects(segment).shouldBumpBulkInsertEpoch).toBe(true)

    const noop = applyPasteIntent({
      intent: { kind: 'noop', reason: 'empty' },
      setInputValue: () => {},
    })
    expect(derivePasteEffects(noop).shouldBumpBulkInsertEpoch).toBe(false)
  })

  test('nextBulkInsertEpochForPaste only increments when effect says so', () => {
    expect(nextBulkInsertEpochForPaste(1, true)).toBe(2)
    expect(nextBulkInsertEpochForPaste(1, false)).toBe(1)
  })
})
