import { describe, expect, test } from 'bun:test'
import { runChatPasteFlow } from './chat-controller'

describe('runChatPasteFlow', () => {
  test('prefers non-empty event text and skips clipboard image fallback', async () => {
    const trace: string[] = []

    const outcome = await runChatPasteFlow({
      eventText: 'pasted-text',
      readClipboardText: () => {
        trace.push('readClipboardText')
        return 'clipboard-text'
      },
      addClipboardImage: async () => {
        trace.push('addClipboardImage')
        return true
      },
      addImageFromFilePath: async () => {
        trace.push('addImageFromFilePath')
        return false
      },
      inlinePastePillCharLimit: 1000,
      insertText: () => trace.push('insertText'),
      insertPasteSegment: () => trace.push('insertPasteSegment'),
    })

    expect(outcome).toBe('text-inline')
    expect(trace).toEqual(['addImageFromFilePath', 'insertText'])
  })

  test('uses event text before clipboard fallback and checks image path before text insertion', async () => {
    const trace: string[] = []

    const outcome = await runChatPasteFlow({
      eventText: '/tmp/image.png',
      readClipboardText: () => {
        trace.push('readClipboardText')
        return 'clipboard-text'
      },
      addClipboardImage: async () => {
        trace.push('addClipboardImage')
        return false
      },
      addImageFromFilePath: async (text) => {
        trace.push(`addImageFromFilePath:${text}`)
        return true
      },
      inlinePastePillCharLimit: 1000,
      insertText: () => trace.push('insertText'),
      insertPasteSegment: () => trace.push('insertPasteSegment'),
    })

    expect(outcome).toBe('pasted-image-path')
    expect(trace).toEqual([
      'addImageFromFilePath:/tmp/image.png',
    ])
  })

  test('falls back to clipboard text when event text is empty and only checks clipboard image when still empty', async () => {
    const trace: string[] = []

    const outcome = await runChatPasteFlow({
      eventText: '',
      readClipboardText: () => {
        trace.push('readClipboardText')
        return ''
      },
      addClipboardImage: async () => {
        trace.push('addClipboardImage')
        return false
      },
      addImageFromFilePath: async () => {
        trace.push('addImageFromFilePath')
        return false
      },
      inlinePastePillCharLimit: 1000,
      insertText: () => trace.push('insertText'),
      insertPasteSegment: () => trace.push('insertPasteSegment'),
    })

    expect(outcome).toBe('empty')
    expect(trace).toEqual(['readClipboardText', 'addClipboardImage'])
  })

  test('falls back to clipboard image when both event and clipboard text are empty', async () => {
    const trace: string[] = []

    const outcome = await runChatPasteFlow({
      eventText: '',
      readClipboardText: () => {
        trace.push('readClipboardText')
        return ''
      },
      addClipboardImage: async () => {
        trace.push('addClipboardImage')
        return true
      },
      addImageFromFilePath: async () => {
        trace.push('addImageFromFilePath')
        return false
      },
      inlinePastePillCharLimit: 1000,
      insertText: () => trace.push('insertText'),
      insertPasteSegment: () => trace.push('insertPasteSegment'),
    })

    expect(outcome).toBe('clipboard-image')
    expect(trace).toEqual(['readClipboardText', 'addClipboardImage'])
  })

  test('routes text payloads to inline vs segment insertion by length', async () => {
    const shortTrace: string[] = []
    const shortOutcome = await runChatPasteFlow({
      eventText: 'hello',
      readClipboardText: () => '',
      addClipboardImage: async () => false,
      addImageFromFilePath: async () => false,
      inlinePastePillCharLimit: 10,
      insertText: (text) => shortTrace.push(`insertText:${text}`),
      insertPasteSegment: (text) => shortTrace.push(`insertPasteSegment:${text}`),
    })
    expect(shortOutcome).toBe('text-inline')
    expect(shortTrace).toEqual(['insertText:hello'])

    const longTrace: string[] = []
    const longOutcome = await runChatPasteFlow({
      eventText: 'this is long',
      readClipboardText: () => '',
      addClipboardImage: async () => false,
      addImageFromFilePath: async () => false,
      inlinePastePillCharLimit: 4,
      insertText: (text) => longTrace.push(`insertText:${text}`),
      insertPasteSegment: (text) => longTrace.push(`insertPasteSegment:${text}`),
    })
    expect(longOutcome).toBe('text-segment')
    expect(longTrace).toEqual(['insertPasteSegment:this is long'])
  })
})
