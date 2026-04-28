import { describe, expect, test } from 'vitest'
import { serializeCanonicalTurn, type CanonicalTrace } from '../src/projections/canonical-xml'
import { YIELD_USER, YIELD_INVOKE } from '@magnitudedev/xml-act'
import { buildResolvedToolSet } from '../src/tools/resolved-toolset'
import { getAgentDefinition, getAgentSlot } from '../src/agents/registry'

const mockConfigState = {
  bySlot: {
    lead: { providerId: 'openai', modelId: 'gpt-5', hardCap: 100000, softCap: 80000 },
    worker: { providerId: 'openai', modelId: 'gpt-5-mini', hardCap: 100000, softCap: 80000 },
  },
} as const

const leadToolSet = buildResolvedToolSet(
  getAgentDefinition('lead'),
  mockConfigState,
  getAgentSlot('lead'),
)

const withYield = (s: string) => `${s}\n${YIELD_USER}`

describe('serializeCanonicalTurn unified format', () => {
  test('think blocks are serialized as <magnitude:think about="..."> tags', () => {
    const trace: CanonicalTrace = { lenses: null, thinkBlocks: [{ about: 'turn', content: 'plan' }], messages: [], toolCalls: [], yieldTarget: 'user' }
    expect(serializeCanonicalTurn(trace, leadToolSet)).toBe(`<magnitude:think about="turn">\nplan\n</magnitude:think>\n${YIELD_USER}`)
  })

  test('think blocks with null about use "think" as name', () => {
    const trace: CanonicalTrace = { lenses: null, thinkBlocks: [{ about: null, content: 'plan' }], messages: [], toolCalls: [], yieldTarget: 'user' }
    expect(serializeCanonicalTurn(trace, leadToolSet)).toBe(`<magnitude:think about="think">\nplan\n</magnitude:think>\n${YIELD_USER}`)
  })

  test('empty think blocks are omitted', () => {
    const trace: CanonicalTrace = { lenses: null, thinkBlocks: [{ about: null, content: '   ' }], messages: [], toolCalls: [], yieldTarget: 'user' }
    expect(serializeCanonicalTurn(trace, leadToolSet)).toBe(YIELD_USER)
  })

  test('messages use <magnitude:message to="..."> syntax', () => {
    const trace: CanonicalTrace = { lenses: null, thinkBlocks: [], messages: [{ text: 'hi', destination: { kind: 'user' } }], toolCalls: [], yieldTarget: 'user' }
    expect(serializeCanonicalTurn(trace, leadToolSet)).toBe(withYield('<magnitude:message to="user">\nhi\n</magnitude:message>'))
  })

  test('tool calls use <magnitude:invoke tool="..."> syntax with <magnitude:parameter> children', () => {
    const trace: CanonicalTrace = { lenses: null, thinkBlocks: [], messages: [], toolCalls: [{ tagName: 'shell', input: { command: 'ls' }, query: null }], yieldTarget: 'user' }
    expect(serializeCanonicalTurn(trace, leadToolSet)).toBe(withYield('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>'))
  })

  test('tool calls with filter include <magnitude:filter> child', () => {
    const trace: CanonicalTrace = { lenses: null, thinkBlocks: [], messages: [], toolCalls: [{ tagName: 'shell', input: { command: 'ls' }, query: '$.stdout' }], yieldTarget: 'user' }
    expect(serializeCanonicalTurn(trace, leadToolSet)).toBe(withYield('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n<magnitude:filter>\n$.stdout\n</magnitude:filter>\n</magnitude:invoke>'))
  })

  test('tool calls with no input emit self-closing invoke', () => {
    const trace: CanonicalTrace = { lenses: null, thinkBlocks: [], messages: [], toolCalls: [{ tagName: 'noop', input: {}, query: null }], yieldTarget: 'user' }
    expect(serializeCanonicalTurn(trace, leadToolSet)).toBe(withYield('<magnitude:invoke tool="noop"/>'))
  })

  test('message text trimming', () => {
    const trace: CanonicalTrace = { lenses: null, thinkBlocks: [], messages: [{ text: '\n\nhey', destination: { kind: 'user' } }], toolCalls: [], yieldTarget: 'user' }
    expect(serializeCanonicalTurn(trace, leadToolSet)).toBe(withYield('<magnitude:message to="user">\nhey\n</magnitude:message>'))
  })

  test("yieldTarget 'invoke' emits yield_invoke", () => {
    const trace: CanonicalTrace = { lenses: null, thinkBlocks: [], messages: [{ text: 'next', destination: { kind: 'user' } }], toolCalls: [], yieldTarget: 'invoke' }
    const xml = serializeCanonicalTurn(trace, leadToolSet)
    expect(xml).toBe(`<magnitude:message to="user">\nnext\n</magnitude:message>\n${YIELD_INVOKE}`)
    expect(xml).not.toContain(YIELD_USER)
    expect(xml).toContain(YIELD_INVOKE)
  })

  test('lenses rendered as magnitude:think about="..." blocks', () => {
    const trace: CanonicalTrace = { lenses: [{ name: 'turn', content: 'my plan' }], thinkBlocks: [], messages: [], toolCalls: [], yieldTarget: 'user' }
    expect(serializeCanonicalTurn(trace, leadToolSet)).toBe(`<magnitude:think about="turn">\nmy plan\n</magnitude:think>\n${YIELD_USER}`)
  })
})
