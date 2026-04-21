import { describe, expect, test } from 'bun:test'
import { serializeCanonicalTurn, type CanonicalTrace } from '../canonical-xml'

const withYield = (s: string) => `${s}\n<idle/>`

describe('serializeCanonicalTurn unified format', () => {
  test('think only', () => {
    const trace: CanonicalTrace = { lenses: null, thinkBlocks: [{ about: null, content: 'plan' }], messages: [], toolCalls: [], turnDecision: 'idle' }
    expect(serializeCanonicalTurn(trace)).toBe('\n<idle/>')
  })

  test('messages are top-level (no comms wrapper)', () => {
    const trace: CanonicalTrace = { lenses: null, thinkBlocks: [], messages: [{ text: 'hi', destination: { kind: 'user' } }], toolCalls: [], turnDecision: 'idle' }
    expect(serializeCanonicalTurn(trace)).toBe(withYield('<message>hi</message>'))
  })

  test('tools are top-level (no actions wrapper)', () => {
    const trace: CanonicalTrace = { lenses: null, thinkBlocks: [], messages: [], toolCalls: [{ tagName: 'shell', input: { command: 'ls' }, query: '.' }], turnDecision: 'idle' }
    expect(serializeCanonicalTurn(trace)).toBe(withYield('<shell observe=".">{"command":"ls"}</shell>'))
  })

  test('full turn preserves ordering', () => {
    const trace: CanonicalTrace = {
      lenses: null,
      thinkBlocks: [{ about: null, content: 'plan' }],
      messages: [{ text: 'done', destination: { kind: 'user' } }],
      toolCalls: [{ tagName: 'shell', input: { command: 'ls' }, query: '.' }],
      turnDecision: 'idle',
    }
    expect(serializeCanonicalTurn(trace)).toBe(
      withYield('\n<message>done</message>\n<shell observe=".">{"command":"ls"}</shell>')
    )
  })

  test('message text trimming', () => {
    const trace: CanonicalTrace = { lenses: null, thinkBlocks: [], messages: [{ text: '\n\nhey', destination: { kind: 'user' } }], toolCalls: [], turnDecision: 'idle' }
    expect(serializeCanonicalTurn(trace)).toBe(withYield('<message>hey</message>'))
  })

  test("turnDecision 'continue' emits continue turn control tag", () => {
    const trace: CanonicalTrace = { lenses: null, thinkBlocks: [], messages: [{ text: 'next', destination: { kind: 'user' } }], toolCalls: [], turnDecision: 'continue' }
    const xml = serializeCanonicalTurn(trace)
    expect(xml).toBe('<message>next</message>\n<continue/>')
    expect(xml).not.toContain('<idle/>')
    expect(xml).toContain('<continue/>')
  })
})
