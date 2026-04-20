
/**
 * Tests for MactBindingValidator
 */

import { describe, it, expect } from 'vitest'
import { Schema } from '@effect/schema'
import { validateMactBinding } from '../mact-binding-validator'

// Test schemas
const ReadSchema = Schema.Struct({
  path: Schema.String,
  offset: Schema.optional(Schema.Number),
  limit: Schema.optional(Schema.Number),
})

const WriteSchema = Schema.Struct({
  path: Schema.String,
  content: Schema.String,
})

const ShellSchema = Schema.Struct({
  command: Schema.String,
  timeout: Schema.optional(Schema.Number),
})

const ComplexOptionsSchema = Schema.Struct({
  id: Schema.String,
  options: Schema.Struct({
    type: Schema.String,
    recursive: Schema.optional(Schema.Boolean),
  }),
})

const ArrayParamSchema = Schema.Struct({
  items: Schema.Array(Schema.Struct({
    name: Schema.String,
    value: Schema.Number,
  })),
})

describe('validateMactBinding', () => {
  it('validates simple scalar parameters', () => {
    const binding = {
      type: 'mact' as const,
      tag: 'read',
      parameters: [
        { name: 'path', field: 'path', type: 'scalar' as const },
        { name: 'offset', field: 'offset', type: 'scalar' as const },
        { name: 'limit', field: 'limit', type: 'scalar' as const },
      ],
    }

    const schema = validateMactBinding('read', binding, ReadSchema.ast)
    
    expect(schema.selfClosing).toBe(false)
    expect(schema.parameters.get('path')).toEqual({
      type: 'string',
      required: true,
      fieldPath: 'path',
    })
    expect(schema.parameters.get('offset')).toEqual({
      type: 'number',
      required: false,
      fieldPath: 'offset',
    })
  })

  it('validates self-closing binding (no parameters)', () => {
    const binding = {
      type: 'mact' as const,
      tag: 'goBack',
      parameters: [],
    }

    const schema = validateMactBinding('goBack', binding, Schema.Struct({}).ast)
    
    expect(schema.selfClosing).toBe(true)
    expect(schema.parameters.size).toBe(0)
  })

  it('validates nested field paths', () => {
    const binding = {
      type: 'mact' as const,
      tag: 'agent-create',
      parameters: [
        { name: 'id', field: 'id', type: 'scalar' as const },
        { name: 'type', field: 'options.type', type: 'scalar' as const },
        { name: 'recursive', field: 'options.recursive', type: 'scalar' as const },
      ],
    }

    const schema = validateMactBinding('agent-create', binding, ComplexOptionsSchema.ast)
    
    expect(schema.parameters.get('type')).toEqual({
      type: 'string',
      required: true,
      fieldPath: 'options.type',
    })
    expect(schema.parameters.get('recursive')).toEqual({
      type: 'boolean',
      required: false,
      fieldPath: 'options.recursive',
    })
  })

  it('validates JSON parameter for array type', () => {
    const binding = {
      type: 'mact' as const,
      tag: 'array-test',
      parameters: [
        { name: 'items', field: 'items', type: 'json' as const },
      ],
    }

    const schema = validateMactBinding('array-test', binding, ArrayParamSchema.ast)
    
    expect(schema.parameters.get('items')).toEqual({
      type: 'json',
      required: true,
      fieldPath: 'items',
    })
  })

  it('throws for non-existent field path', () => {
    const binding = {
      type: 'mact' as const,
      tag: 'read',
      parameters: [
        { name: 'path', field: 'nonexistent', type: 'scalar' as const },
      ],
    }

    expect(() => validateMactBinding('read', binding, ReadSchema.ast)).toThrow(
      "Binding error on <|invoke:read>: parameter field 'nonexistent' does not exist in the schema"
    )
  })

  it('throws for scalar parameter on non-scalar field', () => {
    const binding = {
      type: 'mact' as const,
      tag: 'read',
      parameters: [
        { name: 'items', field: 'items', type: 'scalar' as const },
      ],
    }

    expect(() => validateMactBinding('read', binding, ArrayParamSchema.ast)).toThrow(
      "Binding error on <|invoke:read>: parameter 'items' field 'items' has type 'array' — must be scalar"
    )
  })

  it('throws for JSON parameter on scalar field', () => {
    const binding = {
      type: 'mact' as const,
      tag: 'write',
      parameters: [
        { name: 'path', field: 'path', type: 'json' as const },
      ],
    }

    expect(() => validateMactBinding('write', binding, WriteSchema.ast)).toThrow(
      "Binding error on <|invoke:write>: parameter 'path' field 'path' has type 'string' — must be object or array for JSON parameter"
    )
  })

  it('throws for duplicate parameter names', () => {
    const binding = {
      type: 'mact' as const,
      tag: 'read',
      parameters: [
        { name: 'path', field: 'path', type: 'scalar' as const },
        { name: 'path', field: 'offset', type: 'scalar' as const },
      ],
    }

    expect(() => validateMactBinding('read', binding, ReadSchema.ast)).toThrow(
      "Binding error on <|invoke:read>: parameter name 'path' is declared multiple times"
    )
  })

  it('throws for duplicate field mappings', () => {
    const binding = {
      type: 'mact' as const,
      tag: 'read',
      parameters: [
        { name: 'path1', field: 'path', type: 'scalar' as const },
        { name: 'path2', field: 'path', type: 'scalar' as const },
      ],
    }

    expect(() => validateMactBinding('read', binding, ReadSchema.ast)).toThrow(
      "Binding error on <|invoke:read>: field 'path' is mapped by multiple parameters"
    )
  })
})
