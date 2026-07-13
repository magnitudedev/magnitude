import { describe, expect, it } from 'vitest'
import { ParseResult, Schema } from 'effect'
import {
  deriveStreamingSchema,
  type StreamingSchema,
  type StreamingSchemaResult,
} from '../../ai/src/streaming/streaming-schema'
import { createStreamingFieldParser } from '../../ai/src/streaming/field-parser'
import { JsonValueSchema } from '@magnitudedev/utils/schema'
import { webSearchTool } from '../src/tools/web-search'
import type { ParsedValue } from '../../ai/src/streaming/types'

const num = (value: string, state: 'complete' | 'incomplete' = 'complete'): ParsedValue => ({
  _tag: 'number',
  value,
  state,
})

const str = (value: string, state: 'complete' | 'incomplete' = 'complete'): ParsedValue => ({
  _tag: 'string',
  value,
  state,
})

const bool = (value: boolean): ParsedValue => ({
  _tag: 'boolean',
  value,
  state: 'complete',
})

const obj = (
  entries: Array<[string, ParsedValue]>,
  state: 'complete' | 'incomplete' = 'complete',
): ParsedValue => ({
  _tag: 'object',
  entries,
  state,
})

const arr = (
  items: ParsedValue[],
  state: 'complete' | 'incomplete' = 'complete',
): ParsedValue => ({
  _tag: 'array',
  items,
  state,
})

function decode<A>(streamingSchema: StreamingSchema<A>, value: ParsedValue) {
  return Schema.decodeUnknownEither(streamingSchema)(value)
}

function expectRight<A>(
  result:
    | { readonly _tag: 'Right'; readonly right: StreamingSchemaResult<A> }
    | { readonly _tag: 'Left'; readonly left: unknown },
): StreamingSchemaResult<A> {
  expect(result._tag).toBe('Right')
  if (result._tag === 'Left') throw new Error(String(result.left))
  return result.right
}

function expectLeft(
  result:
    | { readonly _tag: 'Right'; readonly right: unknown }
    | { readonly _tag: 'Left'; readonly left: ParseResult.ParseError },
): ParseResult.ParseError {
  expect(result._tag).toBe('Left')
  if (result._tag === 'Right') throw new Error(`Expected decode failure, got ${JSON.stringify(result.right)}`)
  return result.left
}

function format(error: ParseResult.ParseError) {
  return ParseResult.ArrayFormatter.formatErrorSync(error)
}

describe('deriveStreamingSchema', () => {
  it('returns a real Effect schema that skips incomplete scalars and validates completed scalars', () => {
    const streamingSchema = deriveStreamingSchema(Schema.Number.pipe(Schema.between(1, 10)))

    expect(expectRight(decode(streamingSchema, num('1e', 'incomplete')))).toEqual({ _tag: 'Incomplete' })

    const issues = format(expectLeft(decode(streamingSchema, num('11'))))
    expect(issues[0]?.message).not.toContain('Expected string')
  })

  it('validates completed object children before requiring the whole object to be complete', () => {
    const streamingSchema = deriveStreamingSchema(Schema.Struct({
      score: Schema.Number.pipe(Schema.between(1, 10)),
      summary: Schema.String,
    }))

    expect(expectRight(
      decode(streamingSchema, obj([['score', num('1')]], 'incomplete')),
    )).toEqual({ _tag: 'Incomplete' })
  })

  it('runs container validation when an object completes', () => {
    const streamingSchema = deriveStreamingSchema(Schema.Struct({
      score: Schema.Number,
      summary: Schema.String,
    }))

    const issues = format(expectLeft(decode(streamingSchema, obj([['score', num('1')]]))))
    expect(issues[0]?.path).toEqual(['summary'])
  })

  it('validates record values through derived index schemas', () => {
    const streamingSchema = deriveStreamingSchema(Schema.Record({
      key: Schema.String,
      value: Schema.Number.pipe(Schema.between(1, 10)),
    }))

    const issues = format(expectLeft(decode(streamingSchema, obj([['alpha', num('11')]], 'incomplete'))))
    expect(issues[0]?.path).toEqual(['alpha'])
    expect(issues[0]?.message).not.toContain('Expected string')
  })

  it('uses union child schemas for partial child validation and the full union schema for completed containers', () => {
    const streamingSchema = deriveStreamingSchema(Schema.Union(
      Schema.Struct({ kind: Schema.Literal('score'), value: Schema.Number }),
      Schema.Struct({ kind: Schema.Literal('flag'), value: Schema.Boolean }),
    ))

    expect(expectRight(decode(streamingSchema, obj([
      ['kind', str('score')],
      ['value', bool(true)],
    ], 'incomplete')))).toEqual({ _tag: 'Incomplete' })

    const issues = format(expectLeft(decode(streamingSchema, obj([
      ['kind', str('score')],
      ['value', bool(true)],
    ]))))
    expect(issues[0]?.path[0]).toBeDefined()
  })

  it('rejects duplicate object keys before JSON conversion can erase them', () => {
    const streamingSchema = deriveStreamingSchema(Schema.Struct({
      score: Schema.Number,
    }))

    const issues = format(expectLeft(decode(streamingSchema, obj([
      ['score', num('1')],
      ['score', num('2')],
    ], 'incomplete'))))

    expect(issues[0]).toEqual({
      _tag: 'Type',
      path: ['score'],
      message: 'Duplicate object key "score"',
    })
  })

  it('decodes completed root containers and nullable roots', () => {
    const arraySchema = deriveStreamingSchema(Schema.Array(Schema.Number))
    expect(expectRight(decode(arraySchema, arr([num('1'), num('2')])))).toEqual({
      _tag: 'Complete',
      value: [1, 2],
    })

    const nullSchema = deriveStreamingSchema(Schema.NullOr(Schema.Number))
    expect(expectRight(decode(nullSchema, { _tag: 'null', state: 'complete' }))).toEqual({
      _tag: 'Complete',
      value: null,
    })
  })

  it('derives recursive JSON schemas without overflowing', () => {
    const streamingSchema = deriveStreamingSchema(JsonValueSchema)

    expect(expectRight(decode(streamingSchema, arr([
      num('1'),
      obj([['nested', arr([str('ok')])]]),
    ])))).toEqual({
      _tag: 'Complete',
      value: [1, { nested: ['ok'] }],
    })
  })

  it('preserves final validation for recursive schemas', () => {
    type NumberTree = readonly (number | NumberTree)[]
    const NumberTreeSchema: Schema.Schema<NumberTree> = Schema.suspend((): Schema.Schema<NumberTree> =>
      Schema.Array(Schema.Union(Schema.Number, NumberTreeSchema)) as Schema.Schema<NumberTree>
    )
    const streamingSchema = deriveStreamingSchema(NumberTreeSchema)

    const issues = format(expectLeft(decode(streamingSchema, arr([str('nope')]))))
    expect(issues[0]?.message).toContain('Expected number')
  })

  it('constructs a streaming parser for the web_search input schema', () => {
    const parser = createStreamingFieldParser(webSearchTool.definition.inputSchema)

    parser.push('{"query":"effect schema","schema":{"type":"object","properties":{"name":{"type":"string"}}}}')
    parser.end()

    expect(parser.valid).toBe(true)
  })
})
