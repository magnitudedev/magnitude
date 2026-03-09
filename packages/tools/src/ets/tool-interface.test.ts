import { describe, test, expect } from 'bun:test'
import { Schema } from '@effect/schema'
import { createTool } from '../tool'
import { generateToolInterface, generateToolGroupInterface } from './tool-interface'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeTool(config: {
  name: string
  description?: string
  inputSchema: Schema.Schema.AnyNoContext
  outputSchema?: Schema.Schema.AnyNoContext
  errorSchema?: Schema.Schema.AnyNoContext
  argMapping?: string[]
}) {
  return createTool({
    name: config.name,
    description: config.description ?? '',
    inputSchema: config.inputSchema,
    outputSchema: config.outputSchema ?? Schema.Void,
    errorSchema: config.errorSchema,
    argMapping: config.argMapping,
    execute: () => { throw new Error('not used') },
  })
}

const ItemSchema = Schema.Struct({
  name: Schema.String,
  value: Schema.Number,
}).annotations({ identifier: 'Item' })

const NotFoundError = Schema.Struct({
  _tag: Schema.Literal('NotFoundError'),
  message: Schema.String,
})

const RateLimitError = Schema.Struct({
  _tag: Schema.Literal('RateLimitError'),
  retryAfter: Schema.Number,
})

// ---------------------------------------------------------------------------
// generateToolInterface — extractCommon option
// ---------------------------------------------------------------------------

describe('generateToolInterface — extractCommon', () => {
  const tool = makeTool({
    name: 'getItem',
    description: 'Get an item',
    inputSchema: Schema.Struct({ name: Schema.String }),
    outputSchema: ItemSchema,
    argMapping: ['name'],
  })

  test('extractCommon: true (default) — extracts named types', () => {
    const result = generateToolInterface(tool, 'store')
    expect(result.referencedEntities).toContain('Item')
    expect(result.entityDefinitions.length).toBeGreaterThan(0)
    expect(result.signature).toContain('Item')
    // The signature should use the type reference, not inline the struct
    expect(result.signature).not.toContain('name: string;\n')
  })

  test('extractCommon: false — inlines all types', () => {
    const result = generateToolInterface(tool, 'store', undefined, { extractCommon: false })
    expect(result.referencedEntities).toEqual([])
    expect(result.entityDefinitions).toEqual([])
    // The signature should inline the struct fields
    expect(result.signature).toContain('name: string')
    expect(result.signature).toContain('value: number')
  })
})

// ---------------------------------------------------------------------------
// generateToolInterface — showErrors option
// ---------------------------------------------------------------------------

describe('generateToolInterface — showErrors', () => {
  const tool = makeTool({
    name: 'getItem',
    description: 'Get an item',
    inputSchema: Schema.Struct({ name: Schema.String }),
    outputSchema: Schema.String,
    errorSchema: NotFoundError,
    argMapping: ['name'],
  })

  test('showErrors: true (default) — includes @throws', () => {
    const result = generateToolInterface(tool, 'store')
    expect(result.errorTypes).toContain('store.NotFoundError')
    expect(result.signature).toContain('@throws {store.NotFoundError}')
  })

  test('showErrors: false — omits errors entirely', () => {
    const result = generateToolInterface(tool, 'store', undefined, { showErrors: false })
    expect(result.errorTypes).toEqual([])
    expect(result.signature).not.toContain('@throws')
    expect(result.signature).not.toContain('NotFoundError')
  })
})

// ---------------------------------------------------------------------------
// generateToolInterface — combined options
// ---------------------------------------------------------------------------

describe('generateToolInterface — combined options', () => {
  const tool = makeTool({
    name: 'getItem',
    description: 'Get an item',
    inputSchema: Schema.Struct({ name: Schema.String }),
    outputSchema: ItemSchema,
    errorSchema: NotFoundError,
    argMapping: ['name'],
  })

  test('extractCommon: false + showErrors: false', () => {
    const result = generateToolInterface(tool, 'store', undefined, {
      extractCommon: false,
      showErrors: false,
    })
    expect(result.referencedEntities).toEqual([])
    expect(result.entityDefinitions).toEqual([])
    expect(result.errorTypes).toEqual([])
    // Inlined output type
    expect(result.signature).toContain('name: string')
    expect(result.signature).toContain('value: number')
    // No error annotations
    expect(result.signature).not.toContain('@throws')
  })
})

// ---------------------------------------------------------------------------
// generateToolGroupInterface — useNamespace option
// ---------------------------------------------------------------------------

describe('generateToolGroupInterface — useNamespace', () => {
  const tool1 = makeTool({
    name: 'getItem',
    description: 'Get an item',
    inputSchema: Schema.Struct({ name: Schema.String }),
    outputSchema: ItemSchema,
    errorSchema: NotFoundError,
    argMapping: ['name'],
  })

  const tool2 = makeTool({
    name: 'listItems',
    description: 'List all items',
    inputSchema: Schema.Struct({}),
    outputSchema: Schema.Array(ItemSchema),
  })

  test('useNamespace: true (default) — wraps in declare namespace', () => {
    const result = generateToolGroupInterface('store', [tool1, tool2])
    expect(result).toContain('declare namespace store {')
    expect(result).toContain('}')
    // Entity definitions should be outside the namespace
    expect(result).toContain('type Item =')
  })

  test('useNamespace: false — flat functions with group-prefixed names', () => {
    const result = generateToolGroupInterface('store', [tool1, tool2], { useNamespace: false })
    expect(result).not.toContain('declare namespace')
    // Functions should have group-prefixed names
    expect(result).toContain('function store.getItem')
    expect(result).toContain('function store.listItems')
  })
})

// ---------------------------------------------------------------------------
// generateToolGroupInterface — showErrors option
// ---------------------------------------------------------------------------

describe('generateToolGroupInterface — showErrors', () => {
  const tool = makeTool({
    name: 'getItem',
    description: 'Get an item',
    inputSchema: Schema.Struct({ name: Schema.String }),
    outputSchema: Schema.String,
    errorSchema: Schema.Union(NotFoundError, RateLimitError),
    argMapping: ['name'],
  })

  test('showErrors: true (default) — includes error class declarations', () => {
    const result = generateToolGroupInterface('api', [tool])
    expect(result).toContain('class NotFoundError')
    expect(result).toContain('class RateLimitError')
    expect(result).toContain('@throws')
  })

  test('showErrors: false — omits all error declarations and annotations', () => {
    const result = generateToolGroupInterface('api', [tool], { showErrors: false })
    expect(result).not.toContain('class NotFoundError')
    expect(result).not.toContain('class RateLimitError')
    expect(result).not.toContain('@throws')
  })
})

// ---------------------------------------------------------------------------
// generateToolGroupInterface — extractCommon option
// ---------------------------------------------------------------------------

describe('generateToolGroupInterface — extractCommon', () => {
  const tool1 = makeTool({
    name: 'getItem',
    description: 'Get an item',
    inputSchema: Schema.Struct({ name: Schema.String }),
    outputSchema: ItemSchema,
    argMapping: ['name'],
  })

  const tool2 = makeTool({
    name: 'listItems',
    description: 'List all items',
    inputSchema: Schema.Struct({}),
    outputSchema: Schema.Array(ItemSchema),
  })

  test('extractCommon: true (default) — extracts shared types', () => {
    const result = generateToolGroupInterface('store', [tool1, tool2])
    expect(result).toContain('// Types')
    expect(result).toContain('type Item =')
    // Both functions should reference Item, not inline it
    expect(result).toContain('Item')
    expect(result).toContain('Item[]')
  })

  test('extractCommon: false — inlines all types in every function', () => {
    const result = generateToolGroupInterface('store', [tool1, tool2], { extractCommon: false })
    expect(result).not.toContain('// Types')
    expect(result).not.toContain('type Item =')
    // Both functions should have inlined struct bodies
    // getItem return type:
    expect(result).toContain('name: string')
    expect(result).toContain('value: number')
  })
})

// ---------------------------------------------------------------------------
// generateToolGroupInterface — all options combined
// ---------------------------------------------------------------------------

describe('generateToolGroupInterface — all options combined', () => {
  const tool1 = makeTool({
    name: 'getItem',
    description: 'Get an item',
    inputSchema: Schema.Struct({ name: Schema.String }),
    outputSchema: ItemSchema,
    errorSchema: NotFoundError,
    argMapping: ['name'],
  })

  const tool2 = makeTool({
    name: 'listItems',
    description: 'List all items',
    inputSchema: Schema.Struct({}),
    outputSchema: Schema.Array(ItemSchema),
  })

  test('extractCommon: false, showErrors: false, useNamespace: false', () => {
    const result = generateToolGroupInterface('store', [tool1, tool2], {
      extractCommon: false,
      showErrors: false,
      useNamespace: false,
    })
    // No namespace
    expect(result).not.toContain('declare namespace')
    // No extracted types
    expect(result).not.toContain('// Types')
    expect(result).not.toContain('type Item =')
    // No errors
    expect(result).not.toContain('class NotFoundError')
    expect(result).not.toContain('@throws')
    // Functions present with group-prefixed names and inlined types
    expect(result).toContain('function store.getItem')
    expect(result).toContain('function store.listItems')
    expect(result).toContain('name: string')
    expect(result).toContain('value: number')
  })

  test('extractCommon: true, showErrors: true, useNamespace: true (all defaults)', () => {
    const result = generateToolGroupInterface('store', [tool1, tool2], {
      extractCommon: true,
      showErrors: true,
      useNamespace: true,
    })
    // Namespace present
    expect(result).toContain('declare namespace store {')
    // Extracted types
    expect(result).toContain('type Item =')
    // Error declarations inside namespace
    expect(result).toContain('class NotFoundError')
    expect(result).toContain('@throws')
  })
})

// ---------------------------------------------------------------------------
// Inlining works recursively through nested types
// ---------------------------------------------------------------------------

describe('extractCommon: false — recursive inlining', () => {
  const AddressSchema = Schema.Struct({
    street: Schema.String,
    city: Schema.String,
  }).annotations({ identifier: 'Address' })

  const PersonSchema = Schema.Struct({
    name: Schema.String,
    address: AddressSchema,
  }).annotations({ identifier: 'Person' })

  const tool = makeTool({
    name: 'getPerson',
    description: 'Get person',
    inputSchema: Schema.Struct({ id: Schema.String }),
    outputSchema: PersonSchema,
    argMapping: ['id'],
  })

  test('nested named types are fully inlined', () => {
    const result = generateToolInterface(tool, 'api', undefined, { extractCommon: false })
    expect(result.referencedEntities).toEqual([])
    expect(result.entityDefinitions).toEqual([])
    // The output type should contain the full nested structure
    expect(result.signature).toContain('street: string')
    expect(result.signature).toContain('city: string')
    expect(result.signature).toContain('name: string')
  })

  test('nested named types are extracted when extractCommon: true', () => {
    const result = generateToolInterface(tool, 'api')
    expect(result.referencedEntities).toContain('Person')
    // Should use type references, not inline
    expect(result.signature).not.toContain('street: string')
  })

  test('group-level nested inlining', () => {
    const result = generateToolGroupInterface('api', [tool], {
      extractCommon: false,
      showErrors: false,
      useNamespace: false,
    })
    expect(result).not.toContain('type Person')
    expect(result).not.toContain('type Address')
    expect(result).toContain('street: string')
    expect(result).toContain('city: string')
  })
})

// ---------------------------------------------------------------------------
// Array of named types inlined correctly
// ---------------------------------------------------------------------------

describe('extractCommon: false — arrays of named types', () => {
  const tool = makeTool({
    name: 'listItems',
    description: 'List items',
    inputSchema: Schema.Struct({}),
    outputSchema: Schema.Array(ItemSchema),
  })

  test('Array<NamedType> is inlined as array of struct', () => {
    const result = generateToolInterface(tool, 'store', undefined, { extractCommon: false })
    expect(result.referencedEntities).toEqual([])
    expect(result.entityDefinitions).toEqual([])
    // Should be an inlined array, not Item[]
    expect(result.signature).toContain('name: string')
    expect(result.signature).toContain('value: number')
    expect(result.signature).toContain('[]')
  })
})
