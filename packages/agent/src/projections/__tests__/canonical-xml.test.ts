import { describe, expect, test } from 'bun:test'
import type { XmlTagBinding } from '@magnitudedev/xml-act'
import { serializeCanonicalTurn, type CanonicalTrace } from '../canonical-xml'

const bindings = new Map<string, XmlTagBinding>([
  ['shell', { tag: 'shell', body: 'command' }],
  ['read', { tag: 'read', attributes: [{ field: 'path', attr: 'path' }] }],
  ['write', { tag: 'write', attributes: [{ field: 'path', attr: 'path' }], body: 'content' }],
])
const emptyBindings = new Map<string, XmlTagBinding>()
const withYield = (s: string) => `${s}\n<yield/>`

describe('serializeCanonicalTurn unified format', () => {
  test('think only', () => {
    const trace: CanonicalTrace = { lenses: null, thinkBlocks: [{ about: null, content: 'plan' }], messages: [], toolCalls: [], turnDecision: 'yield' }
    expect(serializeCanonicalTurn(trace, emptyBindings)).toBe('<think>plan</think>\n<yield/>')
  })

  test('messages are top-level (no comms wrapper)', () => {
    const trace: CanonicalTrace = { lenses: null, thinkBlocks: [], messages: [{ text: 'hi' }], toolCalls: [], turnDecision: 'yield' }
    expect(serializeCanonicalTurn(trace, emptyBindings)).toBe(withYield('<message>hi</message>'))
  })

  test('tools are top-level (no actions wrapper)', () => {
    const trace: CanonicalTrace = { lenses: null, thinkBlocks: [], messages: [], toolCalls: [{ tagName: 'shell', input: { command: 'ls' }, query: '.' }], turnDecision: 'yield' }
    expect(serializeCanonicalTurn(trace, bindings)).toBe(withYield('<shell observe=".">ls</shell>'))
  })

  test('full turn preserves ordering', () => {
    const trace: CanonicalTrace = {
      lenses: null,
      thinkBlocks: [{ about: null, content: 'plan' }],
      messages: [{ text: 'done' }],
      toolCalls: [{ tagName: 'shell', input: { command: 'ls' }, query: '.' }],
      turnDecision: 'yield',
    }
    expect(serializeCanonicalTurn(trace, bindings)).toBe(
      withYield('<think>plan</think>\n<message>done</message>\n<shell observe=".">ls</shell>')
    )
  })

  test('message text trimming', () => {
    const trace: CanonicalTrace = { lenses: null, thinkBlocks: [], messages: [{ text: '\n\nhey' }], toolCalls: [], turnDecision: 'yield' }
    expect(serializeCanonicalTurn(trace, emptyBindings)).toBe(withYield('<message>hey</message>'))
  })
})