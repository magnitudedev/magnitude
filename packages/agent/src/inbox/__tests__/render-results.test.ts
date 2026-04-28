import { describe, expect, test } from 'vitest'
import type { ContentPart } from '../../content'
import { formatResults } from '../render-results'

describe('formatResults', () => {
  test('formats observed image result with wrapper tags around inner content only', () => {
    const output = formatResults([
      {
        kind: 'tool_observation',
        toolName: 'view',
        toolCallId: 'tc-view-1',
        content: [
          {
            type: 'image',
            base64: 'dGVzdA==',
            mediaType: 'image/png',
            width: 100,
            height: 100,
          },
        ],
      },
    ], true)

    expect(output).toEqual([
      { type: 'text', text: '\n<view>' },
      { type: 'image', base64: 'dGVzdA==', mediaType: 'image/png', width: 100, height: 100 },
      { type: 'text', text: '</view>' },
    ] satisfies ContentPart[])
  })

  test('keeps runtime execution errors unchanged when no correct tool shape is present', () => {
    const output = formatResults([
      {
        kind: 'tool_error',
        toolName: 'read',
        status: 'error',
        message: 'Failed to read does-not-exist.txt',
      },
    ], true)

    expect(output).toEqual([
      {
        type: 'text',
        text: '\n<tool name="read"><error>Failed to read does-not-exist.txt</error></tool>',
      },
    ] satisfies ContentPart[])
  })

  test('formats no-tools-or-messages notice as plain text result content', () => {
    const output = formatResults([
      {
        kind: 'no_tools_or_messages',
      },
    ], true)

    expect(output).toEqual([
      {
        type: 'text',
        text: '\n(no tools or messages were used this turn)',
      },
    ] satisfies ContentPart[])
  })
})
