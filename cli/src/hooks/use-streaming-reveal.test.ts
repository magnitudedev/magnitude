import { describe, expect, test } from 'vitest'
import React from 'react'
import { act, create } from 'react-test-renderer'
import { useStreamingReveal } from './use-streaming-reveal'

type RevealResult = ReturnType<typeof useStreamingReveal>

function Harness({
  content,
  isStreaming,
  initialDisplayedLength,
  onState,
}: {
  content: string
  isStreaming: boolean
  initialDisplayedLength?: number
  onState: (value: RevealResult) => void
}) {
  const result = useStreamingReveal(content, isStreaming, undefined, initialDisplayedLength)
  onState(result)
  return null
}

describe('useStreamingReveal', () => {
  test('starts from provided initialDisplayedLength when mounting during active streaming', () => {
    let state!: RevealResult
    const content = 'abcdefghij'

    act(() => {
      create(React.createElement(Harness, {
        content,
        isStreaming: true,
        initialDisplayedLength: 7,
        onState: (value: RevealResult) => {
          state = value
        },
      }))
    })

    expect(state.displayedContent).toBe('abcdefg')
    expect(state.isCatchingUp).toBe(true)
  })

  test('defaults to empty reveal when mounting during active streaming without initialDisplayedLength', () => {
    let state!: RevealResult
    const content = 'abcdefghij'

    act(() => {
      create(React.createElement(Harness, {
        content,
        isStreaming: true,
        onState: (value: RevealResult) => {
          state = value
        },
      }))
    })

    expect(state.displayedContent).toBe('')
    expect(state.isCatchingUp).toBe(true)
  })
})
