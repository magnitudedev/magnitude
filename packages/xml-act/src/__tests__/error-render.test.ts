import { describe, it, expect } from 'vitest'
import { renderParseError } from '../presentation/error-render'
import type { StructuralParseErrorEvent, ToolParseErrorEvent } from '../types'

describe('renderParseError', () => {
  it('renders InvalidMagnitudeOpen at top level', () => {
    const event: StructuralParseErrorEvent = {
      _tag: 'StructuralParseError',
      error: {
        _tag: 'InvalidMagnitudeOpen',
        tagName: 'magnitude:foo',
        parentTagName: undefined as any,
        raw: '<magnitude:foo>',
        detail: '',
        primarySpan: { start: { offset: 55, line: 2, col: 1 }, end: { offset: 70, line: 2, col: 16 } },
      },
    }
    const response = '<magnitude:message to="user">Hello</magnitude:message>\n<magnitude:foo>bar</magnitude:foo>\n<magnitude:yield_user/>'
    const result = renderParseError(event, response)
    expect(result).toContain('<parse_error>')
    expect(result).toContain('</parse_error>')
    expect(result).toContain('Unrecognized tag <magnitude:foo> at the top level.')
    expect(result).toContain('2|<magnitude:foo>bar</magnitude:foo>')
    expect(result).toContain('Hints:')
    expect(result).toContain('Only valid magnitude tags are recognized at the top level')
  })

  it('renders InvalidMagnitudeOpen inside invoke', () => {
    const event: StructuralParseErrorEvent = {
      _tag: 'StructuralParseError',
      error: {
        _tag: 'InvalidMagnitudeOpen',
        tagName: 'magnitude:message',
        parentTagName: 'magnitude:invoke',
        raw: '<magnitude:message>',
        detail: '',
        primarySpan: { start: { offset: 31, line: 2, col: 1 }, end: { offset: 50, line: 2, col: 20 } },
      },
    }
    const response = '<magnitude:invoke tool="shell">\n<magnitude:message>oops</magnitude:message>\n</magnitude:invoke>'
    const result = renderParseError(event, response)
    expect(result).toContain('Invalid tag <magnitude:message> inside tool call.')
    expect(result).toContain('2|<magnitude:message>oops</magnitude:message>')
  })

  it('renders StrayCloseTag', () => {
    const event: StructuralParseErrorEvent = {
      _tag: 'StructuralParseError',
      error: {
        _tag: 'StrayCloseTag',
        tagName: 'magnitude:foo',
        detail: '',
        primarySpan: { start: { offset: 0, line: 1, col: 1 }, end: { offset: 16, line: 1, col: 17 } },
      },
    }
    const response = '</magnitude:foo>\n<magnitude:yield_user/>'
    const result = renderParseError(event, response)
    expect(result).toContain('Unexpected close </magnitude:foo> with no matching open.')
    expect(result).toContain('1|</magnitude:foo>')
  })

  it('renders DuplicateParameter tool error with tool section', () => {
    const event: ToolParseErrorEvent = {
      _tag: 'ToolParseError',
      error: {
        _tag: 'DuplicateParameter',
        toolCallId: 'test-1',
        tagName: 'shell',
        parameterName: 'command',
        detail: '',
      },
      toolCallId: 'test-1',
      tagName: 'shell',
      toolName: 'shell',
      group: 'fs',
      correctToolShape: '<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">...</magnitude:parameter>\n</magnitude:invoke>',
    }
    const response = '<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n<magnitude:parameter name="command">pwd</magnitude:parameter>\n</magnitude:invoke>'
    const result = renderParseError(event, response)
    expect(result).toContain("Duplicate parameter 'command' for tool 'shell'.")
    expect(result).toContain('Tool: shell')
    expect(result).toContain('Expected:')
    expect(result).toContain('Each parameter may only appear once')
  })

  it('renders fallback when anchor not found', () => {
    const event: StructuralParseErrorEvent = {
      _tag: 'StructuralParseError',
      error: {
        _tag: 'MalformedTag',
        tagName: 'magnitude:invoke',
        detail: '',
      } as any,
    }
    const response = 'line 1\nline 2\nline 3'
    const result = renderParseError(event, response)
    expect(result).toContain('<parse_error>')
    expect(result).toContain('</parse_error>')
  })
})
