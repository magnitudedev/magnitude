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
    ], true)

    expect(output).toEqual([
      { type: 'text', text: '\n<view observe=".">' },
      { type: 'image', base64: 'dGVzdA==', mediaType: 'image/png', width: 100, height: 100 },
      { type: 'text', text: '</view>' },
    ] satisfies ContentPart[])
  })

  test('formats tool parse errors with parse_error presentation', () => {
    const output = formatResults([
      {
        kind: 'tool_parse_error',
        event: {
          _tag: 'ToolParseError',
          toolCallId: 'tc-read',
          tagName: 'read',
          toolName: 'read',
          group: 'default',
          correctToolShape: '',
          error: {
            _tag: 'MissingRequiredField',
            toolCallId: 'tc-read',
            tagName: 'read',
            parameterName: 'path',
            detail: 'missing required parameter "path".',
          },
        },
        rawResponse: '<magnitude:invoke tool="read">\n</magnitude:invoke>',
      },
    ], true)

    expect(output).toHaveLength(1)
    expect(output[0]).toEqual({ type: 'text', text: expect.any(String) } satisfies ContentPart)
    const text = (output[0] as Extract<ContentPart, { type: 'text' }>).text
    expect(text).toContain('<parse_error>')
    expect(text).toContain("Missing required parameter 'path' for tool 'read'.")
    expect(text).toContain('1|<magnitude:invoke tool="read">')
    expect(text).toContain('2|</magnitude:invoke>')
    expect(text).toContain('Tool: read')
    expect(text).toContain('Hints:')
    expect(text).toContain('- Include all required parameters before closing the tool call.')
  })

  test('formats structural parse errors with parse_error presentation', () => {
    const output = formatResults([
      {
        kind: 'structural_parse_error',
        event: {
          _tag: 'StructuralParseError',
          error: {
            _tag: 'StrayCloseTag',
            tagName: 'magnitude:message',
            detail: '',
          },
        },
        rawResponse: '<magnitude:message to="parent">Hi</magnitude:message>\n</magnitude:message>\n<magnitude:yield_user/>',
      },
    ], true)

    expect(output).toHaveLength(1)
    expect(output[0]).toEqual({ type: 'text', text: expect.any(String) } satisfies ContentPart)
    const text = (output[0] as Extract<ContentPart, { type: 'text' }>).text
    expect(text).toContain('<parse_error>')
    expect(text).toContain('Unexpected close </magnitude:message> with no matching open.')
    expect(text).toContain('1|<magnitude:message to="parent">Hi</magnitude:message>')
    expect(text).toContain('2|</magnitude:message>')
    expect(text).toContain('Hints:')
    expect(text).toContain('- A close tag can only appear after its corresponding open tag.')
    expect(text).toContain('- Use <magnitude:escape> for literal close-tag text.')
  })

  test('keeps runtime execution errors unchanged when no correct tool shape is present', () => {
    const output = formatResults([
      {
        kind: 'tool_error',
        tagName: 'read',
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
