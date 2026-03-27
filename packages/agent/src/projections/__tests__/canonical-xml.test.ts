import { describe, expect, test } from 'bun:test'
import type { XmlTagBinding } from '@magnitudedev/xml-act'
import { serializeCanonicalTurn, type CanonicalTrace } from '../canonical-xml'

const bindings = new Map<string, XmlTagBinding>([
  ['shell', { body: 'command' }],
  ['read', { attributes: [{ field: 'path', attr: 'path' }] }],
  ['write', { attributes: [{ field: 'path', attr: 'path' }], body: 'content' }],
  ['edit', { attributes: [{ field: 'path', attr: 'path' }], childTags: [{ tag: 'old', field: 'old' }, { tag: 'new', field: 'new' }] }],
  ['items-tool', { children: [{ field: 'items', tag: 'item', attributes: [{ field: 'k', attr: 'k' }], body: 'v' }] }],
  ['record-tool', { childRecord: { field: 'vars', keyAttr: 'name', tag: 'var' } }],
  ['attr-order', { attributes: [{ field: 'z', attr: 'z' }, { field: 'a', attr: 'a' }], body: 'content' }],
])
const emptyBindings = new Map<string, XmlTagBinding>()
const withYield = (s: string) => `${s}\n<yield/>`

describe('serializeCanonicalTurn structural variations', () => {
  test('1) think only', () => {
    const trace: CanonicalTrace = {
      lenses: null, thinkBlocks: [{ about: null, content: 'plan' }], messages: [], toolCalls: [], turnDecision: 'yield' }
    expect(serializeCanonicalTurn(trace, emptyBindings)).toBe('<think>plan</think>\n<yield/>')
  })

  test('2) messages only', () => {
    const trace: CanonicalTrace = {
      lenses: null, thinkBlocks: [], messages: [{ dest: 'user', text: 'hi' }], toolCalls: [], turnDecision: 'yield' }
    expect(serializeCanonicalTurn(trace, emptyBindings)).toBe(withYield('<comms>\n<message to="user">hi</message>\n</comms>'))
  })

  test('3) tools only', () => {
    const trace: CanonicalTrace = {
      lenses: null, thinkBlocks: [], messages: [], toolCalls: [{ tagName: 'shell', input: { command: 'ls' }, query: '.' }], turnDecision: 'yield' }
    expect(serializeCanonicalTurn(trace, bindings)).toBe(withYield('<actions>\n<shell observe=".">ls</shell>\n</actions>'))
  })

  test('4) full turn (think + messages + observed tools)', () => {
    const trace: CanonicalTrace = {
      lenses: null,
      thinkBlocks: [{ about: null, content: 'plan' }],
      messages: [{ dest: 'user', text: 'done' }],
      toolCalls: [{ tagName: 'shell', input: { command: 'ls' }, query: '.' }],
      turnDecision: 'yield',
    }
    const expected = [
      '<think>plan</think>',
      '<comms>\n<message to="user">done</message>\n</comms>',
      '<actions>\n<shell observe=".">ls</shell>\n</actions>',
    ].join('\n')
    expect(serializeCanonicalTurn(trace, bindings)).toBe(withYield(expected))
  })

  test('5) multiple messages', () => {
    const trace: CanonicalTrace = {
      lenses: null,
      thinkBlocks: [],
      messages: [{ dest: 'user', text: 'a' }, { dest: 'user', text: 'b' }],
      toolCalls: [],
      turnDecision: 'yield',
    }
    expect(serializeCanonicalTurn(trace, bindings)).toBe(
      withYield('<comms>\n<message to="user">a</message>\n<message to="user">b</message>\n</comms>'),
    )
  })

  test('6) multiple tool calls', () => {
    const trace: CanonicalTrace = {
      lenses: null,
      thinkBlocks: [],
      messages: [],
      toolCalls: [{ tagName: 'shell', input: { command: 'ls' }, query: '.' }, { tagName: 'read', input: { path: 'a.ts' }, query: '.' }],
      turnDecision: 'yield',
    }
    expect(serializeCanonicalTurn(trace, bindings)).toBe(
      withYield('<actions>\n<shell observe=".">ls</shell>\n<read observe="." path="a.ts" />\n</actions>'),
    )
  })

  test('7) tool with attributes only (self-closing)', () => {
    const trace: CanonicalTrace = {
      lenses: null, thinkBlocks: [], messages: [], toolCalls: [{ tagName: 'read', input: { path: 'src/a.ts' }, query: '.' }], turnDecision: 'yield' }
    expect(serializeCanonicalTurn(trace, bindings)).toBe(withYield('<actions>\n<read observe="." path="src/a.ts" />\n</actions>'))
  })

  test('8) tool with body only', () => {
    const trace: CanonicalTrace = {
      lenses: null, thinkBlocks: [], messages: [], toolCalls: [{ tagName: 'shell', input: { command: 'echo hi' }, query: '.' }], turnDecision: 'yield' }
    expect(serializeCanonicalTurn(trace, bindings)).toBe(withYield('<actions>\n<shell observe=".">echo hi</shell>\n</actions>'))
  })

  test('9) tool with attributes and body', () => {
    const trace: CanonicalTrace = {
      lenses: null, thinkBlocks: [], messages: [], toolCalls: [{ tagName: 'write', input: { path: 'a.txt', content: 'hello' }, query: '.' }], turnDecision: 'yield' }
    expect(serializeCanonicalTurn(trace, bindings)).toBe(
      withYield('<actions>\n<write observe="." path="a.txt">hello</' + 'write>\n</actions>'),
    )
  })

  test('10) tool with childTags', () => {
    const trace: CanonicalTrace = {
      lenses: null,
      thinkBlocks: [],
      messages: [],
      toolCalls: [{ tagName: 'edit', input: { path: 'a.ts', old: 'x', new: 'y' }, query: '.' }],
      turnDecision: 'yield',
    }
    expect(serializeCanonicalTurn(trace, bindings)).toBe(
      withYield('<actions>\n<edit observe="." path="a.ts"><old>x</old><new>y</new></edit>\n</actions>'),
    )
  })

  test('11) tool with children array', () => {
    const trace: CanonicalTrace = {
      lenses: null,
      thinkBlocks: [],
      messages: [],
      toolCalls: [{ tagName: 'items-tool', input: { items: [{ k: 'k1', v: 'v1' }, { k: 'k2', v: 'v2' }] }, query: '.' }],
      turnDecision: 'yield',
    }
    expect(serializeCanonicalTurn(trace, bindings)).toBe(
      withYield('<actions>\n<items-tool observe="."><item k="k1">v1</item><item k="k2">v2</item></items-tool>\n</actions>'),
    )
  })

  test('12) tool with childRecord', () => {
    const trace: CanonicalTrace = {
      lenses: null, thinkBlocks: [], messages: [], toolCalls: [{ tagName: 'record-tool', input: { vars: { b: '2', a: '1' } }, query: '.' }], turnDecision: 'yield' }
    expect(serializeCanonicalTurn(trace, bindings)).toBe(
      withYield('<actions>\n<record-tool observe="."><var name="a">1</var><var name="b">2</var></record-tool>\n</actions>'),
    )
  })

  test('13) self-closing tool can serialize non-default observe', () => {
    const trace: CanonicalTrace = {
      lenses: null, thinkBlocks: [], messages: [], toolCalls: [{ tagName: 'read', input: { path: 'x.ts' }, query: '//item[1]' }], turnDecision: 'yield' }
    expect(serializeCanonicalTurn(trace, bindings)).toBe(
      withYield('<actions>\n<read observe="//item[1]" path="x.ts" />\n</actions>'),
    )
  })

  test('14) body tool can serialize non-default observe', () => {
    const trace: CanonicalTrace = {
      lenses: null, thinkBlocks: [], messages: [], toolCalls: [{ tagName: 'shell', input: { command: 'pwd' }, query: 'stdout' }], turnDecision: 'yield' }
    expect(serializeCanonicalTurn(trace, bindings)).toBe(
      withYield('<actions>\n<shell observe="stdout">pwd</shell>\n</actions>'),
    )
  })

  test('15) multiple observed tool calls serialize independently', () => {
    const trace: CanonicalTrace = {
      lenses: null,
      thinkBlocks: [],
      messages: [],
      toolCalls: [
        { tagName: 'shell', input: { command: 'ls' }, query: 'stdout' },
        { tagName: 'read', input: { path: 'a.ts' }, query: '.' },
      ],
      turnDecision: 'yield',
    }
    expect(serializeCanonicalTurn(trace, bindings)).toBe(
      withYield('<actions>\n<shell observe="stdout">ls</shell>\n<read observe="." path="a.ts" />\n</actions>'),
    )
  })

  test('16) empty trace', () => {
    const trace: CanonicalTrace = {
      lenses: null, thinkBlocks: [], messages: [], toolCalls: [], turnDecision: 'yield' }
    expect(serializeCanonicalTurn(trace, bindings)).toBe('<yield/>')
  })

  test('17) message does not include artifacts attribute', () => {
    const trace: CanonicalTrace = {
      lenses: null, thinkBlocks: [], messages: [{ dest: 'user', text: 'ready' }], toolCalls: [], turnDecision: 'yield' }
    expect(serializeCanonicalTurn(trace, bindings)).toBe(withYield('<comms>\n<message to="user">ready</message>\n</comms>'))
  })

  test('18) message with non-default dest', () => {
    const trace: CanonicalTrace = {
      lenses: null, thinkBlocks: [], messages: [{ dest: 'parent', text: 'status' }], toolCalls: [], turnDecision: 'yield' }
    expect(serializeCanonicalTurn(trace, bindings)).toBe(withYield('<comms>\n<message to="parent">status</message>\n</comms>'))
  })

  test('19) deterministic attribute ordering', () => {
    const trace: CanonicalTrace = {
      lenses: null, thinkBlocks: [], messages: [], toolCalls: [{ tagName: 'attr-order', input: { z: '2', a: '1', content: 'x' }, query: '.' }], turnDecision: 'yield' }
    expect(serializeCanonicalTurn(trace, bindings)).toBe(
      withYield('<actions>\n<attr-order a="1" observe="." z="2">x</attr-order>\n</actions>'),
    )
  })

  test('20) tool with no binding fallback', () => {
    const trace: CanonicalTrace = {
      lenses: null, thinkBlocks: [], messages: [], toolCalls: [{ tagName: 'unknown-tool', input: { x: 1 }, query: '.' }], turnDecision: 'yield' }
    expect(serializeCanonicalTurn(trace, bindings)).toBe(
      withYield('<actions>\n<unknown-tool observe=".">{"x":1}</unknown-tool>\n</actions>'),
    )
  })

  test('21) message text is trimmed of leading/trailing whitespace', () => {
    const trace: CanonicalTrace = {
      lenses: null, thinkBlocks: [], messages: [{ dest: 'user', text: '\n\nHey Anders! What are you working on today?' }], toolCalls: [], turnDecision: 'yield' }
    expect(serializeCanonicalTurn(trace, emptyBindings)).toBe(
      withYield('<comms>\n<message to="user">Hey Anders! What are you working on today?</message>\n</comms>'),
    )
  })

  test('22) think text is trimmed of leading/trailing whitespace', () => {
    const trace: CanonicalTrace = {
      lenses: null, thinkBlocks: [{ about: null, content: '\n  Simple greeting.\n' }], messages: [], toolCalls: [], turnDecision: 'yield' }
    expect(serializeCanonicalTurn(trace, emptyBindings)).toBe(withYield('<think>Simple greeting.</think>'))
  })

  test('23) bare prose after think normalizes with trimmed text', () => {
    const trace: CanonicalTrace = {
      lenses: null,
      thinkBlocks: [{ about: null, content: '\n  Simple greeting.\n' }],
      messages: [{ dest: 'user', text: '\n\nHey Anders! What are you working on today?' }],
      toolCalls: [],
      turnDecision: 'yield',
    }
    const expected = '<think>Simple greeting.</think>\n<comms>\n<message to="user">Hey Anders! What are you working on today?</message>\n</comms>'
    expect(serializeCanonicalTurn(trace, emptyBindings)).toBe(withYield(expected))
  })
})