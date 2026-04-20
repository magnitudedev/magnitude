
/**
 * Tests for MactInputBuilder
 */

import { describe, it, expect } from 'vitest'
import { buildMactInput, buildMactInputWithCoercion, coerceParameterValue } from '../mact-input-builder'

describe('coerceParameterValue', () => {
  it('coerces string type', () => {
    expect(coerceParameterValue('hello', 'string')).toBe('hello')
  })

  it('coerces number type', () => {
    expect(coerceParameterValue('42', 'number')).toBe(42)
    expect(coerceParameterValue('3.14', 'number')).toBe(3.14)
  })

  it('throws for invalid number', () => {
    expect(() => coerceParameterValue('not-a-number', 'number')).toThrow()
    expect(() => coerceParameterValue('', 'number')).toThrow()
    expect(() => coerceParameterValue('NaN', 'number')).toThrow()
  })

  it('coerces boolean type with various formats', () => {
    expect(coerceParameterValue('true', 'boolean')).toBe(true)
    expect(coerceParameterValue('True', 'boolean')).toBe(true)
    expect(coerceParameterValue('TRUE', 'boolean')).toBe(true)
    expect(coerceParameterValue('1', 'boolean')).toBe(true)
    
    expect(coerceParameterValue('false', 'boolean')).toBe(false)
    expect(coerceParameterValue('False', 'boolean')).toBe(false)
    expect(coerceParameterValue('FALSE', 'boolean')).toBe(false)
    expect(coerceParameterValue('0', 'boolean')).toBe(false)
  })

  it('throws for invalid boolean', () => {
    expect(() => coerceParameterValue('yes', 'boolean')).toThrow()
    expect(() => coerceParameterValue('no', 'boolean')).toThrow()
    expect(() => coerceParameterValue('maybe', 'boolean')).toThrow()
  })

  it('coerces enum type', () => {
    const enumType = { _tag: 'enum' as const, values: ['foo', 'bar', 'baz'] }
    expect(coerceParameterValue('foo', enumType)).toBe('foo')
    expect(coerceParameterValue('bar', enumType)).toBe('bar')
  })

  it('throws for invalid enum value', () => {
    const enumType = { _tag: 'enum' as const, values: ['foo', 'bar'] }
    expect(() => coerceParameterValue('baz', enumType)).toThrow()
  })
})

describe('buildMactInput', () => {
  it('builds input from scalar parameters', () => {
    const binding = {
      type: 'mact' as const,
      tag: 'read',
      parameters: [
        { name: 'path', field: 'path', type: 'scalar' as const },
        { name: 'offset', field: 'offset', type: 'scalar' as const },
      ],
    }

    const parsed = {
      tagName: 'read',
      toolCallId: 'abc123',
      parameters: new Map([
        ['path', { name: 'path', value: '/workspace/file.ts', isComplete: true }],
        ['offset', { name: 'offset', value: '10', isComplete: true }],
      ]),
    }

    const input = buildMactInput(parsed, binding)
    
    expect(input).toEqual({
      path: '/workspace/file.ts',
      offset: '10', // Not coerced in basic buildMactInput
    })
  })

  it('builds input with nested field paths', () => {
    const binding = {
      type: 'mact' as const,
      tag: 'agent-create',
      parameters: [
        { name: 'id', field: 'id', type: 'scalar' as const },
        { name: 'type', field: 'options.type', type: 'scalar' as const },
      ],
    }

    const parsed = {
      tagName: 'agent-create',
      toolCallId: 'abc123',
      parameters: new Map([
        ['id', { name: 'id', value: 'task-1', isComplete: true }],
        ['type', { name: 'type', value: 'planner', isComplete: true }],
      ]),
    }

    const input = buildMactInput(parsed, binding)
    
    expect(input).toEqual({
      id: 'task-1',
      options: {
        type: 'planner',
      },
    })
  })

  it('builds input with JSON parameter', () => {
    const binding = {
      type: 'mact' as const,
      tag: 'config',
      parameters: [
        { name: 'settings', field: 'settings', type: 'json' as const },
      ],
    }

    const parsed = {
      tagName: 'config',
      toolCallId: 'abc123',
      parameters: new Map([
        ['settings', { name: 'settings', value: '{"theme":"dark","fontSize":14}', isComplete: true }],
      ]),
    }

    const input = buildMactInput(parsed, binding)
    
    expect(input).toEqual({
      settings: { theme: 'dark', fontSize: 14 },
    })
  })

  it('skips missing optional parameters', () => {
    const binding = {
      type: 'mact' as const,
      tag: 'read',
      parameters: [
        { name: 'path', field: 'path', type: 'scalar' as const },
        { name: 'offset', field: 'offset', type: 'scalar' as const },
      ],
    }

    const parsed = {
      tagName: 'read',
      toolCallId: 'abc123',
      parameters: new Map([
        ['path', { name: 'path', value: '/workspace/file.ts', isComplete: true }],
        // offset is missing
      ]),
    }

    const input = buildMactInput(parsed, binding)
    
    expect(input).toEqual({
      path: '/workspace/file.ts',
    })
    expect(input.offset).toBeUndefined()
  })

  it('throws for incomplete parameter', () => {
    const binding = {
      type: 'mact' as const,
      tag: 'read',
      parameters: [
        { name: 'path', field: 'path', type: 'scalar' as const },
      ],
    }

    const parsed = {
      tagName: 'read',
      toolCallId: 'abc123',
      parameters: new Map([
        ['path', { name: 'path', value: '/workspace/file.ts', isComplete: false }],
      ]),
    }

    expect(() => buildMactInput(parsed, binding)).toThrow('Parameter \'path\' is incomplete')
  })

  it('throws for invalid JSON in JSON parameter', () => {
    const binding = {
      type: 'mact' as const,
      tag: 'config',
      parameters: [
        { name: 'settings', field: 'settings', type: 'json' as const },
      ],
    }

    const parsed = {
      tagName: 'config',
      toolCallId: 'abc123',
      parameters: new Map([
        ['settings', { name: 'settings', value: 'not-valid-json', isComplete: true }],
      ]),
    }

    expect(() => buildMactInput(parsed, binding)).toThrow('Invalid JSON')
  })
})

describe('buildMactInputWithCoercion', () => {
  it('coerces scalar values according to schema', () => {
    const binding = {
      type: 'mact' as const,
      tag: 'read',
      parameters: [
        { name: 'path', field: 'path', type: 'scalar' as const },
        { name: 'offset', field: 'offset', type: 'scalar' as const },
        { name: 'limit', field: 'limit', type: 'scalar' as const },
      ],
    }

    const parameterSchemas = new Map([
      ['path', { type: 'string' as const, fieldPath: 'path' }],
      ['offset', { type: 'number' as const, fieldPath: 'offset' }],
      ['limit', { type: 'number' as const, fieldPath: 'limit' }],
    ])

    const parsed = {
      tagName: 'read',
      toolCallId: 'abc123',
      parameters: new Map([
        ['path', { name: 'path', value: '/workspace/file.ts', isComplete: true }],
        ['offset', { name: 'offset', value: '10', isComplete: true }],
        ['limit', { name: 'limit', value: '100', isComplete: true }],
      ]),
    }

    const input = buildMactInputWithCoercion(parsed, binding, parameterSchemas)
    
    expect(input).toEqual({
      path: '/workspace/file.ts',
      offset: 10, // Coerced to number
      limit: 100, // Coerced to number
    })
  })

  it('coerces boolean values', () => {
    const binding = {
      type: 'mact' as const,
      tag: 'write',
      parameters: [
        { name: 'path', field: 'path', type: 'scalar' as const },
        { name: 'replaceAll', field: 'replaceAll', type: 'scalar' as const },
      ],
    }

    const parameterSchemas = new Map([
      ['path', { type: 'string' as const, fieldPath: 'path' }],
      ['replaceAll', { type: 'boolean' as const, fieldPath: 'replaceAll' }],
    ])

    const parsed = {
      tagName: 'write',
      toolCallId: 'abc123',
      parameters: new Map([
        ['path', { name: 'path', value: '/workspace/file.ts', isComplete: true }],
        ['replaceAll', { name: 'replaceAll', value: 'true', isComplete: true }],
      ]),
    }

    const input = buildMactInputWithCoercion(parsed, binding, parameterSchemas)
    
    expect(input).toEqual({
      path: '/workspace/file.ts',
      replaceAll: true, // Coerced to boolean
    })
  })
})
