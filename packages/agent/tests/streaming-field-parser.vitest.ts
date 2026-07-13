import { describe, expect, it } from 'vitest'
import { Schema } from 'effect'
import { createStreamingFieldParser } from '@magnitudedev/ai'
import { escalateTool, passTool } from '../src/observer/schema'

function expectValid(parser: { readonly valid: boolean; readonly validationIssue: unknown }) {
  expect(parser.valid).toBe(true)
  expect(parser.validationIssue).toBeNull()
}

function expectInvalid(parser: { readonly valid: boolean; readonly validationIssue: unknown }) {
  expect(parser.valid).toBe(false)
  expect(parser.validationIssue).not.toBeNull()
}

describe('StreamingFieldParser partial validation', () => {
  describe('incomplete scalar fields', () => {
    it('does not validate an incomplete number before the field completes', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        score: Schema.Number.pipe(Schema.between(1, 10)),
      }))

      parser.push('{"score":1')

      expectValid(parser)
    })

    it('does not validate incomplete numeric spellings before the field completes', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        score: Schema.Number,
      }))

      parser.push('{"score":1e')

      expectValid(parser)
    })

    it('does not validate an incomplete string refinement before the field completes', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        summary: Schema.String.pipe(Schema.minLength(3)),
      }))

      parser.push('{"summary":"x')

      expectValid(parser)
    })
  })

  describe('completed scalar fields', () => {
    it('validates a completed number against the real numeric schema while the root object is still partial', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        score: Schema.Number.pipe(Schema.between(1, 10)),
        summary: Schema.String,
      }))

      parser.push('{"score":1,')

      expectValid(parser)
    })

    it('validates a completed boolean against the real boolean schema while the root object is still partial', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        requires_escalation: Schema.Boolean,
        summary: Schema.String,
      }))

      parser.push('{"requires_escalation":true,')

      expectValid(parser)
    })

    it('rejects a completed number that fails its real refinement', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        score: Schema.Number.pipe(Schema.between(1, 10)),
        summary: Schema.String,
      }))

      parser.push('{"score":11,')

      expectInvalid(parser)
      expect(parser.validationIssue?.message).not.toContain('Expected string')
    })

    it('rejects a completed string that fails its real refinement and keeps the failure sticky', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        summary: Schema.String.pipe(Schema.minLength(3)),
        tail: Schema.String,
      }))

      parser.push('{"summary":"x",')

      expectInvalid(parser)
      expect(parser.validationIssue?.path).toEqual(['summary'])

      parser.push('"tail":"ok"}')
      parser.end()

      expectInvalid(parser)
      expect(parser.decoded).toBeNull()
    })
  })

  describe('object completion', () => {
    it('does not require sibling fields while the containing object is incomplete', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        summary: Schema.String,
        score: Schema.Number,
      }))

      parser.push('{"summary":"ok",')

      expectValid(parser)
    })

    it('rejects a completed nested object that is missing one of its required fields', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        config: Schema.Struct({
          name: Schema.String,
          mode: Schema.String,
        }),
        summary: Schema.String,
      }))

      parser.push('{"config":{"name":"observer"},')

      expectInvalid(parser)
      expect(parser.validationIssue?.path).toEqual(['config', 'mode'])
    })

    it('accepts a completed nested object whose completed children satisfy the nested schema', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        config: Schema.Struct({
          name: Schema.String,
          mode: Schema.String,
        }),
        summary: Schema.String,
      }))

      parser.push('{"config":{"name":"observer","mode":"strict"},')

      expectValid(parser)
    })
  })

  describe('array, tuple, and literal completion', () => {
    it('validates completed array number elements against the real element schema', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        scores: Schema.Array(Schema.Number.pipe(Schema.between(1, 10))),
        summary: Schema.String,
      }))

      parser.push('{"scores":[1,')

      expectValid(parser)
    })

    it('rejects completed array number elements that fail the real element refinement', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        scores: Schema.Array(Schema.Number.pipe(Schema.between(1, 10))),
        summary: Schema.String,
      }))

      parser.push('{"scores":[11,')

      expectInvalid(parser)
      expect(parser.validationIssue?.message).not.toContain('Expected string')
    })

    it('validates tuple slots against their real schemas', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        pair: Schema.Tuple(Schema.String, Schema.Number),
        summary: Schema.String,
      }))

      parser.push('{"pair":["score",1],')

      expectValid(parser)
    })

    it('validates completed numeric and boolean literals as numbers and booleans', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        numeric_kind: Schema.Literal(1),
        boolean_kind: Schema.Literal(true),
        summary: Schema.String,
      }))

      parser.push('{"numeric_kind":1,"boolean_kind":true,')

      expectValid(parser)
    })

    it('validates scalar literal unions without coercing numeric and boolean members to strings', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        mode: Schema.Literal('auto', 1, true),
        summary: Schema.String,
      }))

      parser.push('{"mode":1,')

      expectValid(parser)
    })
  })

  describe('root completion and observer tool schemas', () => {
    it('decodes the escalate tool input with a valid justification', () => {
      const input = {
        justification: 'churn',
      }

      const parser = createStreamingFieldParser(escalateTool.definition.inputSchema)

      parser.push('{"justification":"churn"}')
      parser.end()

      expectValid(parser)
      expect(parser.decoded).toEqual(input)
    })

    it('rejects an invalid justification', () => {
      const parser = createStreamingFieldParser(escalateTool.definition.inputSchema)

      parser.push('{"justification":"invalid_value"}')
      parser.end()

      expectInvalid(parser)
      expect(parser.decoded).toBeNull()
    })

    it('decodes the pass tool input (empty object)', () => {
      const parser = createStreamingFieldParser(passTool.definition.inputSchema)

      parser.push('{}')
      parser.end()

      expectValid(parser)
      expect(parser.decoded).toEqual({})
    })
  })

  describe('schema edge cases', () => {
    it('ignores extra struct fields the same way full Effect Schema decoding does', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        summary: Schema.String,
      }))

      parser.push('{"extra":11,')
      expectValid(parser)

      parser.push('"summary":"ok"}')
      parser.end()

      expectValid(parser)
      expect(parser.decoded).toEqual({ summary: 'ok' })
    })

    it('validates record values through index signatures', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        scores: Schema.Record({
          key: Schema.String,
          value: Schema.Number.pipe(Schema.between(1, 10)),
        }),
        summary: Schema.String,
      }))

      parser.push('{"scores":{"alpha":1,')

      expectValid(parser)
    })

    it('rejects invalid record values through index signatures', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        scores: Schema.Record({
          key: Schema.String,
          value: Schema.Number.pipe(Schema.between(1, 10)),
        }),
        summary: Schema.String,
      }))

      parser.push('{"scores":{"alpha":11,')

      expectInvalid(parser)
      expect(parser.validationIssue?.path).toEqual(['scores', 'alpha'])
      expect(parser.validationIssue?.message).not.toContain('Expected string')
    })

    it('validates null against nullable field schemas', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        score: Schema.NullOr(Schema.Number),
        summary: Schema.String,
      }))

      parser.push('{"score":null,')

      expectValid(parser)
    })

    it('rejects null when the completed field schema is not nullable', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        score: Schema.Number,
        summary: Schema.String,
      }))

      parser.push('{"score":null,')

      expectInvalid(parser)
      expect(parser.validationIssue?.path).toEqual(['score'])
    })

    it('does not require optional fields when the root object completes', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        summary: Schema.String,
        score: Schema.optional(Schema.Number),
      }))

      parser.push('{"summary":"ok"}')
      parser.end()

      expectValid(parser)
      expect(parser.decoded).toEqual({ summary: 'ok' })
    })

    it('validates present optional fields with the real inner schema', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        summary: Schema.String,
        score: Schema.optional(Schema.Number),
      }))

      parser.push('{"score":"bad",')

      expectInvalid(parser)
      expect(parser.validationIssue?.path).toEqual(['score'])
    })

    it('validates union fields at completed child and completed container boundaries', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        payload: Schema.Union(
          Schema.Struct({ kind: Schema.Literal('score'), value: Schema.Number }),
          Schema.Struct({ kind: Schema.Literal('flag'), value: Schema.Boolean }),
        ),
        summary: Schema.String,
      }))

      parser.push('{"payload":{"kind":"score","value":1},')

      expectValid(parser)
    })

    it('uses completed union containers to catch cross-field branch mismatches', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        payload: Schema.Union(
          Schema.Struct({ kind: Schema.Literal('score'), value: Schema.Number }),
          Schema.Struct({ kind: Schema.Literal('flag'), value: Schema.Boolean }),
        ),
        summary: Schema.String,
      }))

      parser.push('{"payload":{"kind":"score","value":true},')

      expectInvalid(parser)
      expect(parser.validationIssue?.path[0]).toBe('payload')
    })

    it('rejects duplicate keys once the duplicate field completes', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        score: Schema.Number,
      }))

      parser.push('{"score":1,"score":2,')

      expectInvalid(parser)
      expect(parser.validationIssue).toEqual({
        path: ['score'],
        message: 'Duplicate object key "score"',
      })
    })

    it('fails clearly when the stream ends before the root value completes', () => {
      const parser = createStreamingFieldParser(Schema.Struct({
        summary: Schema.String,
      }))

      parser.push('{"summary":"ok"')
      parser.end()

      expectInvalid(parser)
      expect(parser.decoded).toBeNull()
      expect(parser.validationIssue?.message).toBe('Input ended before the root value completed')
    })

    it('decodes completed root arrays', () => {
      const parser = createStreamingFieldParser(Schema.Array(Schema.Number))

      parser.push('[1,2,3]')
      parser.end()

      expectValid(parser)
      expect(parser.decoded).toEqual([1, 2, 3])
    })

    it('decodes completed nullable root values without treating null as missing decoded input', () => {
      const parser = createStreamingFieldParser(Schema.NullOr(Schema.Number))

      parser.push('null')
      parser.end()

      expectValid(parser)
      expect(parser.decoded).toBeNull()
    })
  })
})
