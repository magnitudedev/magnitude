/**
 * ParameterAccumulator tests
 *
 * Covers:
 * - Scalar parameter streaming and completion
 * - JSON parameter streaming and completion
 * - Mixed scalar and JSON parameters
 * - Incomplete parameters at read time
 * - Dotted parameter paths (nested fields)
 * - Reset behavior
 * - Multiple tool calls (accumulator reset between calls)
 *
 * Uses RuntimeEvent types (ToolInputFieldValue, ToolInputReady) instead of
 * ParseEvent types (ParameterChunk, ParameterComplete) since the accumulator
 * works with RuntimeEvent from the tool-handle layer.
 */

import { describe, it, expect, beforeEach } from 'vitest'
import { Schema } from '@effect/schema'
import { createParameterAccumulator } from '../accumulator'
import { deriveParameters, type ToolSchema } from '../../execution/parameter-schema'
import type { RuntimeEvent } from '../../types'

// =============================================================================
// Test Helpers
// =============================================================================

function makeToolSchema<T>(schema: Schema.Schema<T>): ToolSchema {
  return deriveParameters(schema.ast)
}

function toolInputStarted(toolCallId: string): RuntimeEvent {
  return {
    _tag: 'ToolInputStarted',
    toolCallId,
    tagName: 'test',
    toolName: 'testTool',
    group: 'test',
  }
}

function toolInputFieldValue(toolCallId: string, field: string, value: string | number | boolean): RuntimeEvent {
  return {
    _tag: 'ToolInputFieldValue',
    toolCallId,
    field,
    value,
  }
}

function toolInputReady(toolCallId: string, input: unknown): RuntimeEvent {
  return {
    _tag: 'ToolInputReady',
    toolCallId,
    input,
  }
}

// =============================================================================
// Scalar Parameter Tests
// =============================================================================

describe('scalar parameters', () => {
  const SimpleSchema = Schema.Struct({
    name: Schema.String,
    count: Schema.Number,
  })

  let accumulator: ReturnType<typeof createParameterAccumulator>

  beforeEach(() => {
    const toolSchema = makeToolSchema(SimpleSchema)
    accumulator = createParameterAccumulator(toolSchema, SimpleSchema.ast)
  })

  it('should accumulate string parameter from ToolInputFieldValue', () => {
    accumulator.ingest(toolInputStarted('call-1'))
    accumulator.ingest(toolInputFieldValue('call-1', 'name', 'Hello'))

    const current = accumulator.current
    expect(current.name).toEqual({ isFinal: false, value: 'Hello' })
  })

  it('should mark scalar as complete when ToolInputReady received', () => {
    accumulator.ingest(toolInputStarted('call-1'))
    accumulator.ingest(toolInputFieldValue('call-1', 'name', 'World'))
    accumulator.ingest(toolInputReady('call-1', { name: 'World', count: 0 }))

    const current = accumulator.current
    expect(current.name).toEqual({ isFinal: true, value: 'World' })
  })

  it('should coerce number from string', () => {
    accumulator.ingest(toolInputStarted('call-1'))
    accumulator.ingest(toolInputFieldValue('call-1', 'count', '42'))
    accumulator.ingest(toolInputReady('call-1', { name: '', count: 42 }))

    const current = accumulator.current
    expect(current.count).toEqual({ isFinal: true, value: 42 })
  })

  it('should handle multiple parameters independently', () => {
    accumulator.ingest(toolInputStarted('call-1'))
    accumulator.ingest(toolInputFieldValue('call-1', 'name', 'Test'))
    accumulator.ingest(toolInputFieldValue('call-1', 'count', '123'))
    accumulator.ingest(toolInputReady('call-1', { name: 'Test', count: 123 }))

    const current = accumulator.current
    expect(current.name).toEqual({ isFinal: true, value: 'Test' })
    expect(current.count).toEqual({ isFinal: true, value: 123 })
  })
})

// =============================================================================
// JSON Parameter Tests
// =============================================================================

describe('json parameters', () => {
  const JsonSchema = Schema.Struct({
    items: Schema.Array(Schema.Struct({
      name: Schema.String,
      value: Schema.Number,
    })),
  })

  let accumulator: ReturnType<typeof createParameterAccumulator>

  beforeEach(() => {
    const toolSchema = makeToolSchema(JsonSchema)
    accumulator = createParameterAccumulator(toolSchema, JsonSchema.ast)
  })

  it('should stream JSON parameter from ToolInputFieldValue', () => {
    accumulator.ingest(toolInputStarted('call-1'))
    accumulator.ingest(toolInputFieldValue('call-1', 'items', '[{"name":'))

    const current1 = accumulator.current
    // JSON is incomplete, should have partial structure
    expect(current1.items).toBeDefined()
  })

  it('should complete JSON parameter and produce final value', () => {
    accumulator.ingest(toolInputStarted('call-1'))
    accumulator.ingest(toolInputFieldValue('call-1', 'items', '[{"name":"foo","value":42}]'))
    accumulator.ingest(toolInputReady('call-1', { items: [{ name: 'foo', value: 42 }] }))

    const current = accumulator.current
    expect(current.items).toBeDefined()
    const items = current.items as Array<Record<string, { isFinal: boolean; value: unknown }>>
    expect(items).toHaveLength(1)
    expect(items[0].name).toEqual({ isFinal: true, value: 'foo' })
    expect(items[0].value).toEqual({ isFinal: true, value: 42 })
  })

  it('should handle incomplete JSON mid-stream', () => {
    accumulator.ingest(toolInputStarted('call-1'))
    accumulator.ingest(toolInputFieldValue('call-1', 'items', '[{"name":"incomplete'))

    const current = accumulator.current
    expect(current.items).toBeDefined()
  })
})

// =============================================================================
// Mixed Scalar and JSON Tests
// =============================================================================

describe('mixed scalar and json parameters', () => {
  const MixedSchema = Schema.Struct({
    path: Schema.String,
    items: Schema.Array(Schema.Struct({
      id: Schema.Number,
      name: Schema.String,
    })),
  })

  let accumulator: ReturnType<typeof createParameterAccumulator>

  beforeEach(() => {
    const toolSchema = makeToolSchema(MixedSchema)
    accumulator = createParameterAccumulator(toolSchema, MixedSchema.ast)
  })

  it('should handle scalar and JSON parameters together', () => {
    accumulator.ingest(toolInputStarted('call-1'))
    accumulator.ingest(toolInputFieldValue('call-1', 'path', '/workspace'))
    accumulator.ingest(toolInputFieldValue('call-1', 'items', '[{"id":1,"name":"first"}]'))
    accumulator.ingest(toolInputReady('call-1', { path: '/workspace', items: [{ id: 1, name: 'first' }] }))

    const current = accumulator.current
    expect(current.path).toEqual({ isFinal: true, value: '/workspace' })
    expect(current.items).toBeDefined()
    const items = current.items as Array<Record<string, { isFinal: boolean; value: unknown }>>
    expect(items).toHaveLength(1)
    expect(items[0].id).toEqual({ isFinal: true, value: 1 })
    expect(items[0].name).toEqual({ isFinal: true, value: 'first' })
  })
})

// =============================================================================
// Dotted Parameter Path Tests
// =============================================================================

describe('dotted parameter paths', () => {
  const NestedSchema = Schema.Struct({
    options: Schema.Struct({
      type: Schema.Literal('explorer', 'builder'),
      depth: Schema.Number,
    }),
  })

  let accumulator: ReturnType<typeof createParameterAccumulator>

  beforeEach(() => {
    const toolSchema = makeToolSchema(NestedSchema)
    accumulator = createParameterAccumulator(toolSchema, NestedSchema.ast)
  })

  it('should create nested structure from dotted paths', () => {
    accumulator.ingest(toolInputStarted('call-1'))
    accumulator.ingest(toolInputFieldValue('call-1', 'options.type', 'explorer'))
    accumulator.ingest(toolInputFieldValue('call-1', 'options.depth', '5'))
    accumulator.ingest(toolInputReady('call-1', { options: { type: 'explorer', depth: 5 } }))

    const current = accumulator.current
    expect(current.options).toBeDefined()
    const options = current.options as Record<string, { isFinal: boolean; value: unknown }>
    expect(options.type).toEqual({ isFinal: true, value: 'explorer' })
    expect(options.depth).toEqual({ isFinal: true, value: 5 })
  })
})

// =============================================================================
// Reset Behavior Tests
// =============================================================================

describe('reset behavior', () => {
  const SimpleSchema = Schema.Struct({
    value: Schema.String,
  })

  it('should clear all state on reset', () => {
    const toolSchema = makeToolSchema(SimpleSchema)
    const accumulator = createParameterAccumulator(toolSchema, SimpleSchema.ast)

    accumulator.ingest(toolInputStarted('call-1'))
    accumulator.ingest(toolInputFieldValue('call-1', 'value', 'test'))
    accumulator.reset()

    const current = accumulator.current
    expect(Object.keys(current)).toHaveLength(0)
  })

  it('should handle multiple tool calls with reset', () => {
    const toolSchema = makeToolSchema(SimpleSchema)
    const accumulator = createParameterAccumulator(toolSchema, SimpleSchema.ast)

    // First call
    accumulator.ingest(toolInputStarted('call-1'))
    accumulator.ingest(toolInputFieldValue('call-1', 'value', 'first'))
    accumulator.ingest(toolInputReady('call-1', { value: 'first' }))

    let current = accumulator.current
    expect(current.value).toEqual({ isFinal: true, value: 'first' })

    // Reset and second call
    accumulator.reset()
    accumulator.ingest(toolInputStarted('call-2'))
    accumulator.ingest(toolInputFieldValue('call-2', 'value', 'second'))
    accumulator.ingest(toolInputReady('call-2', { value: 'second' }))

    current = accumulator.current
    expect(current.value).toEqual({ isFinal: true, value: 'second' })
  })
})

// =============================================================================
// Incomplete Parameter Tests
// =============================================================================

describe('incomplete parameters', () => {
  const SimpleSchema = Schema.Struct({
    name: Schema.String,
    count: Schema.Number,
  })

  it('should show incomplete state for parameters not yet completed', () => {
    const toolSchema = makeToolSchema(SimpleSchema)
    const accumulator = createParameterAccumulator(toolSchema, SimpleSchema.ast)

    accumulator.ingest(toolInputStarted('call-1'))
    accumulator.ingest(toolInputFieldValue('call-1', 'name', 'Incom'))
    // Don't send ToolInputReady

    const current = accumulator.current
    expect(current.name).toEqual({ isFinal: false, value: 'Incom' })
    expect(current.count).toBeUndefined()
  })
})

// =============================================================================
// Edge Cases
// =============================================================================

describe('edge cases', () => {
  it('should ignore events when not initialized', () => {
    const SimpleSchema = Schema.Struct({
      value: Schema.String,
    })
    const toolSchema = makeToolSchema(SimpleSchema)
    const accumulator = createParameterAccumulator(toolSchema, SimpleSchema.ast)

    // Send ToolInputFieldValue without ToolInputStarted
    accumulator.ingest(toolInputFieldValue('call-1', 'value', 'test'))

    const current = accumulator.current
    expect(Object.keys(current)).toHaveLength(0)
  })

  it('should ignore unknown parameters', () => {
    const SimpleSchema = Schema.Struct({
      value: Schema.String,
    })
    const toolSchema = makeToolSchema(SimpleSchema)
    const accumulator = createParameterAccumulator(toolSchema, SimpleSchema.ast)

    accumulator.ingest(toolInputStarted('call-1'))
    accumulator.ingest(toolInputFieldValue('call-1', 'unknown', 'test'))

    const current = accumulator.current
    expect(current.value).toBeUndefined()
    expect((current as Record<string, unknown>).unknown).toBeUndefined()
  })
})
