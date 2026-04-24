import { describe, expect, it } from 'vitest'

import { generateInvocationExample } from '../presentation/invocation-example'

describe('generateInvocationExample', () => {
  it('renders required parameters with canonical placeholders', () => {
    const parameters = new Map([
      ['command', { name: 'command', type: 'string' as const, required: true }],
    ])

    expect(generateInvocationExample('shell', parameters)).toBe(
      [
        '<magnitude:invoke tool="shell">',
        '<magnitude:parameter name="command">...</magnitude:parameter>',
        '</magnitude:invoke>',
      ].join('\n'),
    )
  })

  it('renders required parameters before optional parameters', () => {
    const parameters = new Map([
      ['glob', { name: 'glob', type: 'string' as const, required: false }],
      ['pattern', { name: 'pattern', type: 'string' as const, required: true }],
      ['limit', { name: 'limit', type: 'number' as const, required: false }],
    ])

    expect(generateInvocationExample('grep', parameters)).toBe(
      [
        '<magnitude:invoke tool="grep">',
        '<magnitude:parameter name="pattern">...</magnitude:parameter>',
        '<magnitude:parameter name="glob">...</magnitude:parameter> <!-- optional -->',
        '<magnitude:parameter name="limit">123</magnitude:parameter> <!-- optional -->',
        '</magnitude:invoke>',
      ].join('\n'),
    )
  })

  it('renders enum, boolean, and json placeholders', () => {
    const parameters = new Map([
      ['mode', { name: 'mode', type: { _tag: 'enum' as const, values: ['fast', 'slow'] }, required: true }],
      ['recursive', { name: 'recursive', type: 'boolean' as const, required: true }],
      ['payload', { name: 'payload', type: 'json_object' as const, required: true }],
      ['items', { name: 'items', type: 'json_array' as const, required: true }],
    ])

    expect(generateInvocationExample('example', parameters)).toBe(
      [
        '<magnitude:invoke tool="example">',
        '<magnitude:parameter name="mode">fast</magnitude:parameter>',
        '<magnitude:parameter name="recursive">true</magnitude:parameter>',
        '<magnitude:parameter name="payload">{...}</magnitude:parameter>',
        '<magnitude:parameter name="items">[...]</magnitude:parameter>',
        '</magnitude:invoke>',
      ].join('\n'),
    )
  })

  it('omits optional parameters when showOptional is false', () => {
    const parameters = new Map([
      ['pattern', { name: 'pattern', type: 'string' as const, required: true }],
      ['path', { name: 'path', type: 'string' as const, required: false }],
    ])

    expect(generateInvocationExample('grep', parameters, { showOptional: false })).toBe(
      [
        '<magnitude:invoke tool="grep">',
        '<magnitude:parameter name="pattern">...</magnitude:parameter>',
        '</magnitude:invoke>',
      ].join('\n'),
    )
  })

  it('renders zero-parameter tools as self-closing', () => {
    expect(generateInvocationExample('tree', new Map())).toBe('<magnitude:invoke tool="tree"/>')
  })

  it('renders compact output without newlines', () => {
    const parameters = new Map([
      ['command', { name: 'command', type: 'string' as const, required: true }],
      ['limit', { name: 'limit', type: 'number' as const, required: false }],
    ])

    expect(generateInvocationExample('shell', parameters, { compact: true })).toBe(
      '<magnitude:invoke tool="shell"><magnitude:parameter name="command">...</magnitude:parameter><magnitude:parameter name="limit">123</magnitude:parameter> <!-- optional --></magnitude:invoke>',
    )
  })
})
