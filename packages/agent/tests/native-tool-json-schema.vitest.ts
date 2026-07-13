import { describe, expect, it } from 'vitest'
import { Schema } from 'effect'
import { NoInputSchema } from '@magnitudedev/utils/schema'
import { makeNativeToolParametersJsonSchema } from '../../ai/src'
import { webSearchTool } from '../src/tools/web-search'

describe('native tool parameter JSON Schema', () => {
  it('accepts strict no-input object schemas', () => {
    const jsonSchema = makeNativeToolParametersJsonSchema(NoInputSchema)

    expect(jsonSchema.$schema).toBeUndefined()
    expect(jsonSchema.$ref).toBe('#/$defs/NoInput')
  })

  it('rejects non-object tool parameter roots', () => {
    expect(() => makeNativeToolParametersJsonSchema(Schema.String)).toThrow(
      'Native tool parameters must be encoded as a JSON object schema',
    )
  })

  it('generates native parameters for web_search recursive JSON input', () => {
    const jsonSchema = makeNativeToolParametersJsonSchema(webSearchTool.definition.inputSchema)

    expect(jsonSchema.type).toBe('object')
    expect(jsonSchema.$defs).toHaveProperty('JsonValue')
  })
})
