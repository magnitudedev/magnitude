import { describe, expect, test } from 'bun:test'
import { getCommunicationPreview } from './agent-communication-card'

describe('agent communication card preview', () => {
  test('does not overflow when content wraps to 3 visual lines or fewer', () => {
    expect(getCommunicationPreview('abcdef', 10).hasOverflow).toBe(false)
    expect(getCommunicationPreview('abcdef', 2).hasOverflow).toBe(false) // 3 lines exactly
  })

  test('overflows when a single logical line wraps past 3 visual lines', () => {
    const result = getCommunicationPreview('abcdefgh', 2)
    expect(result.hasOverflow).toBe(true)
    expect(result.previewLines).toEqual(['ab', 'cd', 'ef'])
  })

  test('respects explicit newlines as hard breaks in visual wrapping', () => {
    const result = getCommunicationPreview('ab\ncdefgh', 2)
    expect(result.hasOverflow).toBe(true)
    expect(result.previewLines).toEqual(['ab', 'cd', 'ef'])
  })
})