import { describe, expect, test } from 'vitest'
import { serializeCanonicalTurn, type CanonicalTrace } from '../src/projections/canonical-xml'
import { YIELD_USER, YIELD_INVOKE } from '@magnitudedev/xml-act'

const withYield = (s: string) => `${s}\n${YIELD_USER}`

describe('serializeCanonicalTurn unified format', () => {
  test('reason blocks are serialized as <reason about="..."> tags', () => {
    const trace: CanonicalTrace = { lenses: null, reasonBlocks: [{ about: 'turn', content: 'plan' }], messages: [], toolCalls: [], turnDecision: 'idle' }
    expect(serializeCanonicalTurn(trace)).toBe(`<reason about="turn">\nplan\n</reason>\n${YIELD_USER}`)
  })

  test('reason blocks with null about use "reason" as name', () => {
    const trace: CanonicalTrace = { lenses: null, reasonBlocks: [{ about: null, content: 'plan' }], messages: [], toolCalls: [], turnDecision: 'idle' }
    expect(serializeCanonicalTurn(trace)).toBe(`<reason about="reason">\nplan\n</reason>\n${YIELD_USER}`)
  })

  test('empty think blocks are omitted', () => {
    const trace: CanonicalTrace = { lenses: null, reasonBlocks: [{ about: null, content: '   ' }], messages: [], toolCalls: [], turnDecision: 'idle' }
    expect(serializeCanonicalTurn(trace)).toBe(YIELD_USER)
  })

  test('messages use <message to="..."> syntax', () => {
    const trace: CanonicalTrace = { lenses: null, reasonBlocks: [], messages: [{ text: 'hi', destination: { kind: 'user' } }], toolCalls: [], turnDecision: 'idle' }
    expect(serializeCanonicalTurn(trace)).toBe(withYield('<message to="user">\nhi\n</message>'))
  })

  test('tool calls use <invoke tool="..."> syntax with <parameter> children', () => {
    const trace: CanonicalTrace = { lenses: null, reasonBlocks: [], messages: [], toolCalls: [{ tagName: 'shell', input: { command: 'ls' }, query: null }], turnDecision: 'idle' }
    expect(serializeCanonicalTurn(trace)).toBe(withYield('<invoke tool="shell">\n<parameter name="command">ls</parameter>\n</invoke>'))
  })

  test('tool calls with filter include <filter> child', () => {
    const trace: CanonicalTrace = { lenses: null, reasonBlocks: [], messages: [], toolCalls: [{ tagName: 'shell', input: { command: 'ls' }, query: '$.stdout' }], turnDecision: 'idle' }
    expect(serializeCanonicalTurn(trace)).toBe(withYield('<invoke tool="shell">\n<parameter name="command">ls</parameter>\n<filter>\n$.stdout\n</filter>\n</invoke>'))
  })

  test('tool calls with no input emit self-closing invoke', () => {
    const trace: CanonicalTrace = { lenses: null, reasonBlocks: [], messages: [], toolCalls: [{ tagName: 'noop', input: {}, query: null }], turnDecision: 'idle' }
    expect(serializeCanonicalTurn(trace)).toBe(withYield('<invoke tool="noop"/>'))
  })

  test('message text trimming', () => {
    const trace: CanonicalTrace = { lenses: null, reasonBlocks: [], messages: [{ text: '\n\nhey', destination: { kind: 'user' } }], toolCalls: [], turnDecision: 'idle' }
    expect(serializeCanonicalTurn(trace)).toBe(withYield('<message to="user">\nhey\n</message>'))
  })

  test("turnDecision 'continue' emits yield_invoke", () => {
    const trace: CanonicalTrace = { lenses: null, reasonBlocks: [], messages: [{ text: 'next', destination: { kind: 'user' } }], toolCalls: [], turnDecision: 'continue' }
    const xml = serializeCanonicalTurn(trace)
    expect(xml).toBe(`<message to="user">\nnext\n</message>\n${YIELD_INVOKE}`)
    expect(xml).not.toContain(YIELD_USER)
    expect(xml).toContain(YIELD_INVOKE)
  })

  test('lenses rendered as <reason about="..."> blocks', () => {
    const trace: CanonicalTrace = { lenses: [{ name: 'turn', content: 'my plan' }], reasonBlocks: [], messages: [], toolCalls: [], turnDecision: 'idle' }
    expect(serializeCanonicalTurn(trace)).toBe(`<reason about="turn">\nmy plan\n</reason>\n${YIELD_USER}`)
  })
})
