import { describe, expect, it } from 'vitest'
import type { InputValue } from '../types/store'
import {
  applyTextEditWithPastesAndMentions,
  insertMentionSegment,
  reconstituteInputTextWithMentions,
} from './strings'

function emptyInput(text: string): InputValue {
  return {
    text,
    cursorPosition: text.length,
    lastEditDueToNav: false,
    pasteSegments: [],
    mentionSegments: [],
    selectedPasteSegmentId: null,
    selectedMentionSegmentId: null,
  }
}

describe('ordered mention segments', () => {
  it('preserves repeated occurrences with distinct ids and authored spans', () => {
    const first = insertMentionSegment(emptyInput('@a then @a'), {
      path: 'src/a.ts',
      contentType: 'text',
    }, 'first', 0, 2)
    const secondStart = first.text.lastIndexOf('@a')
    const second = insertMentionSegment(first, {
      path: 'src/a.ts',
      contentType: 'text',
    }, 'second', secondStart, secondStart + 2)

    const result = reconstituteInputTextWithMentions(second)
    expect(result.text).toBe('@src/a.ts then @src/a.ts')
    expect(result.mentions.map(mention => ({
      id: mention.id,
      start: mention.start,
      end: mention.end,
    }))).toEqual([
      { id: 'first', start: 0, end: 9 },
      { id: 'second', start: 15, end: 24 },
    ])
  })

  it('removes an edited mention atomically and shifts later occurrences', () => {
    const input: InputValue = {
      ...emptyInput('@a @b'),
      mentionSegments: [
        { id: 'first', path: 'a', contentType: 'text', start: 0, end: 2 },
        { id: 'second', path: 'b', contentType: 'text', start: 3, end: 5 },
      ],
    }

    const result = applyTextEditWithPastesAndMentions(input, 1, 2, 'x')
    expect(result.text).toBe('x @b')
    expect(result.mentionSegments).toEqual([
      { id: 'second', path: 'b', contentType: 'text', start: 2, end: 4 },
    ])
  })
})
