import { describe, expect, test } from 'vitest'
import type { ContentPart } from '../../content'
import { formatResults } from '../render-results'

describe('formatResults', () => {
  test('formats observed image result with wrapper tags around inner content only', () => {
    const output = formatResults(
      [] as const,
      [{
        tagName: 'view',
        toolCallId: 'test-1',
        query: '.',
        content: [
          {
            type: 'image',
            base64: 'dGVzdA==',
            mediaType: 'image/png',
            width: 100,
            height: 100,
          },
        ],
      }],
    )

    expect(output).toEqual([
      { type: 'text', text: '\n<view observe=".">' },
      { type: 'image', base64: 'dGVzdA==', mediaType: 'image/png', width: 100, height: 100 },
      { type: 'text', text: '</view>' },
    ] satisfies ContentPart[])
  })
})
