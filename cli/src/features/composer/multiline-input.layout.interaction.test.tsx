import { act, useState } from 'react'
import { testRender } from '@opentui/react/test-utils'
import type { InputValue } from '@magnitudedev/client-common'
import { describe, expect, test, vi } from 'vitest'

vi.hoisted(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
    .IS_REACT_ACT_ENVIRONMENT = true
})

vi.mock('../../hooks/use-theme', () => ({
  useTheme: () => ({
    muted: '#888888',
    inputFocusedFg: '#ffffff',
    inputFg: '#ffffff',
    info: '#00aaff',
  }),
}))

const { MultilineInput } = await import('./multiline-input')

function inputValue(text: string): InputValue {
  return {
    text,
    cursorPosition: text.length,
    lastEditDueToNav: false,
    pasteSegments: [],
    mentionSegments: [],
  }
}

function lineContaining(frame: string, text: string): number {
  return frame.split('\n').findIndex((line) => line.includes(text))
}

describe('MultilineInput layout synchronization', () => {
  test('expands immediately when Shift+Enter adds a line', async () => {
    function Harness() {
      const [input, setInput] = useState(() => inputValue('one'))
      return (
        <box style={{ flexDirection: 'column' }}>
          <MultilineInput
            value={input.text}
            cursorPosition={input.cursorPosition}
            onChange={setInput}
            onSubmit={() => {}}
            onPaste={() => false}
            shouldBlinkCursor={false}
            maxHeight={5}
          />
          <text>AFTER</text>
        </box>
      )
    }

    const view = await testRender(<Harness />, {
      width: 30,
      height: 8,
      kittyKeyboard: true,
    })

    try {
      await act(async () => view.flush())
      expect(lineContaining(view.captureCharFrame(), 'AFTER')).toBe(1)

      await act(async () => view.mockInput.pressEnter({ shift: true }))
      await act(async () => view.flush())

      const frame = view.captureCharFrame()
      expect(lineContaining(frame, 'one')).toBe(0)
      expect(lineContaining(frame, 'AFTER')).toBe(2)
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })

  test('collapses immediately when a full-height value is cleared', async () => {
    let clear: () => void = () => {
      throw new Error('Harness has not rendered')
    }

    function Harness() {
      const [input, setInput] = useState(() => inputValue('one\ntwo\nthree\nfour\nfive'))
      clear = () => setInput(inputValue(''))

      return (
        <box style={{ flexDirection: 'column' }}>
          <MultilineInput
            value={input.text}
            cursorPosition={input.cursorPosition}
            onChange={setInput}
            onSubmit={() => {}}
            onPaste={() => false}
            shouldBlinkCursor={false}
            maxHeight={5}
          />
          <text>AFTER</text>
        </box>
      )
    }

    const view = await testRender(<Harness />, { width: 30, height: 8 })

    try {
      await act(async () => view.flush())
      expect(lineContaining(view.captureCharFrame(), 'AFTER')).toBe(5)

      await act(async () => clear())
      await act(view.renderOnce)

      expect(lineContaining(view.captureCharFrame(), 'AFTER')).toBe(1)
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })
})
