import { describe, expect, test } from 'vitest'
import type { ContentPart } from '../../content'
import { formatResults } from '../render-results'

describe('formatResults', () => {
  test('formats observed image result with wrapper tags around inner content only', () => {
    const output = formatResults([
      {
        kind: 'tool_observation',
        tagName: 'view',
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
      },
    ])

    expect(output).toEqual([
      { type: 'text', text: '\n<view observe=".">' },
      { type: 'image', base64: 'dGVzdA==', mediaType: 'image/png', width: 100, height: 100 },
      { type: 'text', text: '</view>' },
    ] satisfies ContentPart[])
  })

  test('formats invalid tool input with inline correct tool shape inside error block', () => {
    const output = formatResults([
      {
        kind: 'tool_error',
        tagName: 'read',
        status: 'error',
        message: 'Invalid tool input: missing required attribute "path".',
        correctToolShape: '<read\npath="..."\noffset="..."\nlimit="..."\n/>',
      },
    ])

    expect(output).toEqual([
      {
        type: 'text',
        text: '\n<tool name="read"><error>\nInvalid tool input: missing required attribute "path".\n\nCorrect tool shape:\n<read\npath="..."\noffset="..."\nlimit="..."\n/>\n</error></tool>',
      },
    ] satisfies ContentPart[])
  })

  test('keeps runtime execution errors unchanged when no correct tool shape is present', () => {
    const output = formatResults([
      {
        kind: 'tool_error',
        tagName: 'read',
        status: 'error',
        message: 'Failed to read does-not-exist.txt',
      },
    ])

    expect(output).toEqual([
      {
        type: 'text',
        text: '\n<tool name="read"><error>Failed to read does-not-exist.txt</error></tool>',
      },
    ] satisfies ContentPart[])
  })
})
