import { describe, expect, test } from 'vitest'
import { Option } from 'effect'
import { renderCompactToolCall } from '../render-tool-call'

describe('renderCompactToolCall', () => {
  test('renders self-closing tag when no body', () => {
    expect(renderCompactToolCall({
      toolName: 'read',
      attributes: { path: 'src/auth.ts' },
      body: Option.none(),
    })).toBe('<read path="src/auth.ts"/>')
  })

  test('renders tag with body', () => {
    expect(renderCompactToolCall({
      toolName: 'shell',
      attributes: { exitCode: '0' },
      body: Option.some('npm test'),
    })).toBe('<shell exitCode="0">npm test</shell>')
  })

  test('truncates body at default limit', () => {
    const body = 'x'.repeat(501)
    const output = renderCompactToolCall({
      toolName: 'shell',
      attributes: {},
      body: Option.some(body),
    })

    expect(output).toContain('... (truncated)</shell>')
    expect(output.startsWith('<shell>')).toBe(true)
  })

  test('truncates body at custom limit', () => {
    expect(renderCompactToolCall({
      toolName: 'shell',
      attributes: {},
      body: Option.some('abcdef'),
      maxBodyChars: 3,
    })).toBe('<shell>abc... (truncated)</shell>')
  })

  test('sorts attributes alphabetically', () => {
    expect(renderCompactToolCall({
      toolName: 'tool',
      attributes: { z: '2', a: '1', m: '3' },
      body: Option.none(),
    })).toBe('<tool a="1" m="3" z="2"/>')
  })

  test('renders empty attributes without spaces', () => {
    expect(renderCompactToolCall({
      toolName: 'noop',
      attributes: {},
      body: Option.none(),
    })).toBe('<noop/>')
  })

  test('does not XML-escape values', () => {
    expect(renderCompactToolCall({
      toolName: 'x',
      attributes: { raw: '<tag>&"\'', other: 'a<b' },
      body: Option.some('<body>&"\''),
    })).toBe('<x other="a<b" raw="<tag>&"\'"><body>&"\'</x>')
  })
})
