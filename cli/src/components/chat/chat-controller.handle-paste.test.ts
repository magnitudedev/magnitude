import { describe, expect, test } from 'bun:test'
import { handleChatControllerPaste } from './chat-controller'
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

})
