import { describe, expect, test } from 'bun:test'
import type { XmlTagBinding } from '@magnitudedev/xml-act'
import { serializeCanonicalTurn, type CanonicalTrace } from '../canonical-xml'

const bindings = new Map<string, XmlTagBinding>([
  ['shell', { tag: 'shell', body: 'command' }],
  ['read', { tag: 'read', attributes: [{ field: 'path', attr: 'path' }] }],
  ['write', { tag: 'write', attributes: [{ field: 'path', attr: 'path' }], body: 'content' }],
])
const emptyBindings = new Map<string, XmlTagBinding>()
const withYield = (s: string) => `${s}\n<idle/>`

describe('serializeCanonicalTurn unified format', () => {
  test('think only', () => {
    const trace: CanonicalTrace = { lenses: null, thinkBlocks: [{ about: null, content: 'plan' }], messages: [], toolCalls: [], turnDecision: 'idle' }
    expect(serializeCanonicalTurn(trace, emptyBindings)).toBe('<think>plan</think>\n<idle/>')
  })

  test('messages are top-level (no comms wrapper)', () => {
    const trace: CanonicalTrace = { lenses: null, thinkBlocks: [], messages: [{ text: 'hi', destination: { kind: 'user' } }], toolCalls: [], turnDecision: 'idle' }
    expect(serializeCanonicalTurn(trace, emptyBindings)).toBe(withYield('<message>hi</message>'))
  })

  test('tools are top-level (no actions wrapper)', () => {
    const trace: CanonicalTrace = { lenses: null, thinkBlocks: [], messages: [], toolCalls: [{ tagName: 'shell', input: { command: 'ls' }, query: '.' }], turnDecision: 'idle' }
    expect(serializeCanonicalTurn(trace, bindings)).toBe(withYield('<shell observe=".">ls</shell>'))
  })

  test('full turn preserves ordering', () => {
    const trace: CanonicalTrace = {
      lenses: null,
      thinkBlocks: [{ about: null, content: 'plan' }],
      messages: [{ text: 'done', destination: { kind: 'user' } }],
      toolCalls: [{ tagName: 'shell', input: { command: 'ls' }, query: '.' }],
      turnDecision: 'idle',
    }
    expect(serializeCanonicalTurn(trace, bindings)).toBe(
      withYield('<think>plan</think>\n<message>done</message>\n<shell observe=".">ls</shell>')
    )
  })

  test('message text trimming', () => {
    const trace: CanonicalTrace = { lenses: null, thinkBlocks: [], messages: [{ text: '\n\nhey', destination: { kind: 'user' } }], toolCalls: [], turnDecision: 'idle' }
    expect(serializeCanonicalTurn(trace, emptyBindings)).toBe(withYield('<message>hey</message>'))
  })

  test("turnDecision 'continue' emits no turn-control or legacy tags", () => {
    const trace: CanonicalTrace = { lenses: null, thinkBlocks: [], messages: [{ text: 'next', destination: { kind: 'user' } }], toolCalls: [], turnDecision: 'continue' }
    const xml = serializeCanonicalTurn(trace, emptyBindings)
    expect(xml).toBe('<message>next</message>')
    expect(xml).not.toContain('<idle/>')
    expect(xml).not.toContain('<wait/>')
    expect(xml).not.toContain('<yield/>')
  })
})