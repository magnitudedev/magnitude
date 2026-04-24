import { describe, expect, test } from 'vitest'
import { serializeCanonicalTurn, type CanonicalTrace } from '../src/projections/canonical-xml'
import { YIELD_USER, YIELD_INVOKE } from '@magnitudedev/xml-act'
import { buildResolvedToolSet } from '../src/tools/resolved-toolset'
import { getAgentDefinition, getAgentSlot } from '../src/agents/registry'

const mockConfigState = {
  bySlot: {
    main: { providerId: 'openai', modelId: 'gpt-5', hardCap: 100000, softCap: 80000 },
    background: { providerId: 'openai', modelId: 'gpt-5-mini', hardCap: 100000, softCap: 80000 },
    thinker: { providerId: 'openai', modelId: 'gpt-5', hardCap: 100000, softCap: 80000 },
    title: { providerId: 'openai', modelId: 'gpt-5-mini', hardCap: 100000, softCap: 80000 },
    compact: { providerId: 'openai', modelId: 'gpt-5-mini', hardCap: 100000, softCap: 80000 },
  },
} as const

const leadToolSet = buildResolvedToolSet(
  getAgentDefinition('lead'),
  mockConfigState,
  getAgentSlot('lead'),
)

const withYield = (s: string) => `${s}\n${YIELD_USER}`

describe('serializeCanonicalTurn unified format', () => {
  test('reason blocks are serialized as <magnitude:reason about="..."> tags', () => {
    const trace: CanonicalTrace = { lenses: null, reasonBlocks: [{ about: 'turn', content: 'plan' }], messages: [], toolCalls: [], yieldTarget: 'user' }
    expect(serializeCanonicalTurn(trace, leadToolSet)).toBe(`<magnitude:reason about="turn">\nplan\n</magnitude:reason>\n${YIELD_USER}`)
  })

  test('reason blocks with null about use "reason" as name', () => {
    const trace: CanonicalTrace = { lenses: null, reasonBlocks: [{ about: null, content: 'plan' }], messages: [], toolCalls: [], yieldTarget: 'user' }
    expect(serializeCanonicalTurn(trace, leadToolSet)).toBe(`<magnitude:reason about="reason">\nplan\n</magnitude:reason>\n${YIELD_USER}`)
  })

  test('empty think blocks are omitted', () => {
    const trace: CanonicalTrace = { lenses: null, reasonBlocks: [{ about: null, content: '   ' }], messages: [], toolCalls: [], yieldTarget: 'user' }
    expect(serializeCanonicalTurn(trace, leadToolSet)).toBe(YIELD_USER)
  })

  test('messages use <magnitude:message to="..."> syntax', () => {
    const trace: CanonicalTrace = { lenses: null, reasonBlocks: [], messages: [{ text: 'hi', destination: { kind: 'user' } }], toolCalls: [], yieldTarget: 'user' }
    expect(serializeCanonicalTurn(trace, leadToolSet)).toBe(withYield('<magnitude:message to="user">\nhi\n</magnitude:message>'))
  })

  test('tool calls use <magnitude:invoke tool="..."> syntax with <magnitude:parameter> children', () => {
    const trace: CanonicalTrace = { lenses: null, reasonBlocks: [], messages: [], toolCalls: [{ tagName: 'shell', input: { command: 'ls' }, query: null }], yieldTarget: 'user' }
    expect(serializeCanonicalTurn(trace, leadToolSet)).toBe(withYield('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n</magnitude:invoke>'))
  })

  test('tool calls with filter include <magnitude:filter> child', () => {
    const trace: CanonicalTrace = { lenses: null, reasonBlocks: [], messages: [], toolCalls: [{ tagName: 'shell', input: { command: 'ls' }, query: '$.stdout' }], yieldTarget: 'user' }
    expect(serializeCanonicalTurn(trace, leadToolSet)).toBe(withYield('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">ls</magnitude:parameter>\n<magnitude:filter>\n$.stdout\n</magnitude:filter>\n</magnitude:invoke>'))
  })

  test('tool calls with no input emit self-closing invoke', () => {
    const trace: CanonicalTrace = { lenses: null, reasonBlocks: [], messages: [], toolCalls: [{ tagName: 'noop', input: {}, query: null }], yieldTarget: 'user' }
    expect(serializeCanonicalTurn(trace, leadToolSet)).toBe(withYield('<magnitude:invoke tool="noop"/>'))
  })

  test('message text trimming', () => {
    const trace: CanonicalTrace = { lenses: null, reasonBlocks: [], messages: [{ text: '\n\nhey', destination: { kind: 'user' } }], toolCalls: [], yieldTarget: 'user' }
    expect(serializeCanonicalTurn(trace, leadToolSet)).toBe(withYield('<magnitude:message to="user">\nhey\n</magnitude:message>'))
  })

  test("yieldTarget 'invoke' emits yield_invoke", () => {
    const trace: CanonicalTrace = { lenses: null, reasonBlocks: [], messages: [{ text: 'next', destination: { kind: 'user' } }], toolCalls: [], yieldTarget: 'invoke' }
    const xml = serializeCanonicalTurn(trace, leadToolSet)
    expect(xml).toBe(`<magnitude:message to="user">\nnext\n</magnitude:message>\n${YIELD_INVOKE}`)
    expect(xml).not.toContain(YIELD_USER)
    expect(xml).toContain(YIELD_INVOKE)
  })

  test('lenses rendered as <magnitude:reason about="..."> blocks', () => {
    const trace: CanonicalTrace = { lenses: [{ name: 'turn', content: 'my plan' }], reasonBlocks: [], messages: [], toolCalls: [], yieldTarget: 'user' }
    expect(serializeCanonicalTurn(trace, leadToolSet)).toBe(`<magnitude:reason about="turn">\nmy plan\n</magnitude:reason>\n${YIELD_USER}`)
  })
})
