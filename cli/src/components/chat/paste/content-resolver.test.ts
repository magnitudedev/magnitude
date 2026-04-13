import { describe, expect, test } from 'bun:test'
import { resolvePasteIntent } from './content-resolver'

describe('resolvePasteIntent', () => {
  test('prefers event text over clipboard text', async () => {
    const result = await resolvePasteIntent({
      eventText: 'event',
      readClipboardText: () => 'clipboard',
      tryAddClipboardImage: async () => false,
      tryAddImageFromFilePath: async () => false,
      inlinePastePillCharLimit: 1000,
    })
    expect(result).toEqual({ kind: 'insert-inline-text', text: 'event' })
  })

  test('empty text can resolve clipboard image', async () => {
    const result = await resolvePasteIntent({
      eventText: '',
      readClipboardText: () => '',
      tryAddClipboardImage: async () => true,
      tryAddImageFromFilePath: async () => false,
      inlinePastePillCharLimit: 1000,
    })
    expect(result).toEqual({ kind: 'add-clipboard-image' })
  })

  test('pasted path resolves to path image intent', async () => {
    const result = await resolvePasteIntent({
      eventText: '/tmp/test.png',
      readClipboardText: () => '',
      tryAddClipboardImage: async () => false,
      tryAddImageFromFilePath: async () => true,
      inlinePastePillCharLimit: 1000,
    })
    expect(result).toEqual({ kind: 'add-path-image', rawPath: '/tmp/test.png' })
  })

  test('multi-path payload resolves to path image intent when callback succeeds', async () => {
    const payload = '/Users/me/a.png /Users/me/b.png'
    const result = await resolvePasteIntent({
      eventText: payload,
      readClipboardText: () => '',
      tryAddClipboardImage: async () => false,
      tryAddImageFromFilePath: async () => true,
      inlinePastePillCharLimit: 1000,
    })
    expect(result).toEqual({ kind: 'add-path-image', rawPath: payload })
  })

  test('multi-path payload falls back to text when callback fails', async () => {
    const payload = '/Users/me/a.png /Users/me/b.png'
    const result = await resolvePasteIntent({
      eventText: payload,
      readClipboardText: () => '',
      tryAddClipboardImage: async () => false,
      tryAddImageFromFilePath: async () => false,
      inlinePastePillCharLimit: 1000,
    })
    expect(result).toEqual({ kind: 'insert-inline-text', text: payload })
  })

  test('blocked resolves to blocked noop', async () => {
    const result = await resolvePasteIntent({
      blocked: true,
      eventText: 'x',
      readClipboardText: () => 'y',
      tryAddClipboardImage: async () => false,
      tryAddImageFromFilePath: async () => false,
      inlinePastePillCharLimit: 1000,
    })
    expect(result).toEqual({ kind: 'noop', reason: 'blocked' })
  })
})
