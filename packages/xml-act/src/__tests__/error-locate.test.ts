import { describe, expect, it } from 'vitest'

import { buildSnippet, findErrorLine } from '../presentation/error-locate'

describe('findErrorLine', () => {
  it('finds the 1-indexed line where the anchor first appears', () => {
    const response = [
      '<magnitude:reason>',
      'thinking',
      '</magnitude:reason>',
      '<magnitude:message>Hello</magnitude:message>',
    ].join('\n')

    expect(findErrorLine(response, '<magnitude:message>')).toBe(4)
  })

  it('returns the first matching line when the anchor appears multiple times', () => {
    const response = [
      '<magnitude:message>one</magnitude:message>',
      '<magnitude:message>two</magnitude:message>',
    ].join('\n')

    expect(findErrorLine(response, '<magnitude:message>')).toBe(1)
  })

  it('returns null when the anchor is not found', () => {
    expect(findErrorLine('alpha\nbeta', '<magnitude:message>')).toBeNull()
  })

  it('returns null for empty response text or empty anchor', () => {
    expect(findErrorLine('', 'alpha')).toBeNull()
    expect(findErrorLine('alpha', '')).toBeNull()
  })
})

describe('buildSnippet', () => {
  it('renders a point snippet with one line before and after', () => {
    const response = ['line1', 'line2', 'line3', 'line4'].join('\n')

    expect(buildSnippet(response, 2, 'point')).toBe(['1|line1', '2|line2', '3|line3'].join('\n'))
  })

  it('omits missing before and after lines for point snippets at file edges', () => {
    const response = ['line1', 'line2'].join('\n')

    expect(buildSnippet(response, 1, 'point')).toBe(['1|line1', '2|line2'].join('\n'))
    expect(buildSnippet(response, 2, 'point')).toBe(['1|line1', '2|line2'].join('\n'))
  })

  it('renders a short block snippet from block start through error line plus one after', () => {
    const response = ['a', 'b', 'c', 'd', 'e'].join('\n')

    expect(buildSnippet(response, 4, 'block', 2)).toBe(
      ['2|b', '3|c', '4|d', '5|e'].join('\n'),
    )
  })

  it('caps long block snippets and inserts an ellipsis', () => {
    const response = Array.from({ length: 12 }, (_, i) => `line${i + 1}`).join('\n')

    expect(buildSnippet(response, 11, 'block', 1)).toBe(
      ['1|line1', '2|line2', '...', '10|line10', '11|line11', '12|line12'].join('\n'),
    )
  })

  it('clamps block start to valid bounds', () => {
    const response = ['a', 'b', 'c'].join('\n')

    expect(buildSnippet(response, 2, 'block', 0)).toBe(['1|a', '2|b', '3|c'].join('\n'))
    expect(buildSnippet(response, 2, 'block', 3)).toBe(['2|b', '3|c'].join('\n'))
  })

  it('returns an empty string for empty response text', () => {
    expect(buildSnippet('', 1, 'point')).toBe('')
    expect(buildSnippet('', 1, 'block', 1)).toBe('')
  })
})
