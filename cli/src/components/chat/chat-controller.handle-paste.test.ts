import { describe, expect, test } from 'bun:test'
import { addImageAttachmentsFromPastedText, handleChatControllerPaste } from './chat-controller'
import type { ImageMediaType } from '@magnitudedev/agent'
import type { InputValue } from '../../types/store'

const EMPTY_INPUT: InputValue = {
  text: '',
  cursorPosition: 0,
  lastEditDueToNav: false,
  pasteSegments: [],
  mentionSegments: [],
  selectedPasteSegmentId: null,
  selectedMentionSegmentId: null,
}

function makeMockCandidate(filename: string, mediaType: ImageMediaType = 'image/png') {
  return {
    filename,
    base64: 'ZmFrZQ==',
    mediaType,
    width: 10,
    height: 20,
  }
}

describe('handleChatControllerPaste', () => {
  test('inserts inline text into input state via real chat-controller paste path', async () => {
    let state = EMPTY_INPUT
    const outcome = await handleChatControllerPaste({
      eventText: 'hello',
      addClipboardImage: async () => false,
      addImageFromFilePath: async () => false,
      setInputValue: (updater) => {
        state = updater(state)
      },
    })

    expect(outcome.didInsert).toBe(true)
    expect(outcome.shouldBumpBulkInsertEpoch).toBe(true)
    expect(state.text).toBe('hello')
    expect(state.cursorPosition).toBe(5)
  })

  test('prefers image-path branch and skips text insertion', async () => {
    let state = EMPTY_INPUT
    let imagePathCalls = 0
    const outcome = await handleChatControllerPaste({
      eventText: '/tmp/image.png',
      addClipboardImage: async () => false,
      addImageFromFilePath: async () => {
        imagePathCalls += 1
        return true
      },
      setInputValue: (updater) => {
        state = updater(state)
      },
    })

    expect(outcome.didInsert).toBe(true)
    expect(outcome.shouldBumpBulkInsertEpoch).toBe(false)
    expect(imagePathCalls).toBe(1)
    expect(state.text).toBe('')
  })

  test('real multi-attach callback: at least one valid candidate resolves => attachments added and raw text suppressed', async () => {
    let state = EMPTY_INPUT
    const attachments: Array<{ filename: string }> = []
    const outcome = await handleChatControllerPaste({
      eventText: '/Users/me/a.png /Users/me/b.png',
      addClipboardImage: async () => false,
      addImageFromFilePath: async (rawPasteText) =>
        addImageAttachmentsFromPastedText({
          rawPasteText,
          appendAttachments: (newAttachments) => {
            attachments.push(...newAttachments.map((item) => ({ filename: item.filename })))
          },
          extractCandidates: () => ['/Users/me/a.png', '/Users/me/b.png'],
          readCandidate: async (candidate) =>
            candidate.endsWith('a.png') ? makeMockCandidate('a.png') : makeMockCandidate('b.png'),
          scaleAttachment: async (value) => value,
        }),
      setInputValue: (updater) => {
        state = updater(state)
      },
    })

    expect(outcome.didInsert).toBe(true)
    expect(outcome.shouldBumpBulkInsertEpoch).toBe(false)
    expect(attachments.map((item) => item.filename)).toEqual(['a.png', 'b.png'])
    expect(state.text).toBe('')
  })

  test('real multi-attach callback: zero valid candidates => raw text remains fallback insertion', async () => {
    let state = EMPTY_INPUT
    const payload = '/Users/me/a.png /Users/me/b.png'
    const outcome = await handleChatControllerPaste({
      eventText: payload,
      addClipboardImage: async () => false,
      addImageFromFilePath: async (rawPasteText) =>
        addImageAttachmentsFromPastedText({
          rawPasteText,
          appendAttachments: () => {},
          extractCandidates: () => ['/Users/me/a.png', '/Users/me/b.png'],
          readCandidate: async () => null,
          scaleAttachment: async (value) => value,
        }),
      setInputValue: (updater) => {
        state = updater(state)
      },
    })

    expect(outcome.didInsert).toBe(true)
    expect(outcome.shouldBumpBulkInsertEpoch).toBe(true)
    expect(state.text).toBe(payload)
  })

  test('real multi-attach callback: mixed valid/invalid candidates => valid images attach', async () => {
    const attachments: Array<{ filename: string }> = []
    const result = await addImageAttachmentsFromPastedText({
      rawPasteText: '/Users/me/a.png /Users/me/not-image.txt /Users/me/b.png',
      appendAttachments: (newAttachments) => {
        attachments.push(...newAttachments.map((item) => ({ filename: item.filename })))
      },
      extractCandidates: () => ['/Users/me/a.png', '/Users/me/not-image.txt', '/Users/me/b.png'],
      readCandidate: async (candidate) => {
        if (candidate.endsWith('a.png')) return makeMockCandidate('a.png')
        if (candidate.endsWith('b.png')) return makeMockCandidate('b.png')
        return null
      },
      scaleAttachment: async (value) => value,
    })

    expect(result).toBe(true)
    expect(attachments.map((item) => item.filename)).toEqual(['a.png', 'b.png'])
  })
})
