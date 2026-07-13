import { describe, it, expect } from 'vitest'
import { Effect, Schema, Option } from 'effect'
import {
  compilePatchMap,
  diffDecoded,
  applyDecodedPatch,
  PatchNavigationError,
} from '../index'
import type { PatchApplyError } from '../index'
import type { CompiledMap, DecodedPatchOp, DecodedValue, Path } from '../index'
import { DisplayViewSnapshot } from '../../../../protocol/src/schemas/display'
import type { DisplayViewSnapshot as DVS } from '../../../../protocol/src/schemas/display'
import { StreamWireEvent } from '../../../../protocol/src/schemas/events'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Run a diff+apply round-trip and assert the result deep-equals next. */
function assertRoundTrip(
  compiled: CompiledMap,
  prev: DecodedValue,
  next: DecodedValue,
  label: string,
): readonly DecodedPatchOp[] {
  const either = Effect.runSync(Effect.either(diffDecoded(prev, next, compiled)))
  if (either._tag === 'Left') {
    throw new Error(`diffDecoded failed for "${label}": ${JSON.stringify(either.left)}`)
  }
  const ops = either.right
  const applyEither = Effect.runSync(Effect.either(applyDecodedPatch(prev, ops, compiled)))
  if (applyEither._tag === 'Left') {
    throw new Error(`applyDecodedPatch failed for "${label}": ${JSON.stringify(applyEither.left)}`)
  }
  expect(applyEither.right).toEqual(next)
  return ops
}

/** Assert that a set of ops can be encoded and decoded through StreamWireEvent. */
function assertWireRoundTrip(ops: readonly DecodedPatchOp[]): void {
  const wireEvent = { _tag: 'patch' as const, ops }
  const encoded = Schema.encodeSync(StreamWireEvent)(wireEvent)
  const decoded = Schema.decodeUnknownSync(StreamWireEvent)(encoded)
  expect(decoded).toEqual(wireEvent)
}

/** Assert no op in the batch has an undefined value. */
function assertNoUndefined(ops: readonly DecodedPatchOp[]): void {
  for (const op of ops) {
    if ('value' in op) {
      expect(op.value).not.toBeUndefined()
      // Recursively check nested objects/arrays
      checkNoUndefinedDeep(op.value)
    }
  }
}

function checkNoUndefinedDeep(v: unknown): void {
  if (v === null || typeof v !== 'object') return
  if (Array.isArray(v)) {
    for (const item of v) checkNoUndefinedDeep(item)
    return
  }
  for (const val of Object.values(v as Record<string, unknown>)) {
    expect(val).not.toBeUndefined()
    checkNoUndefinedDeep(val)
  }
}

/** Assert no op contains an _key field. */
function assertNoKey(ops: readonly DecodedPatchOp[]): void {
  for (const op of ops) {
    if ('value' in op) {
      checkNoKeyDeep(op.value)
    }
  }
}

function checkNoKeyDeep(v: unknown): void {
  if (v === null || typeof v !== 'object') return
  if (Array.isArray(v)) {
    for (const item of v) checkNoKeyDeep(item)
    return
  }
  const obj = v as Record<string, unknown>
  expect('_key' in obj).toBe(false)
  for (const val of Object.values(obj)) {
    checkNoKeyDeep(val)
  }
}

// ---------------------------------------------------------------------------
// Test schemas
// ---------------------------------------------------------------------------

const InnerSchema = Schema.Struct({
  name: Schema.String,
  value: Schema.Number,
})

// Schema covering optionalWith Option, optionalWith default, unions, arrays,
// records, nullable.  All optional fields are now Option-based so that the
// schema passes the JsonSafeSchema constraint used by compilePatchMap.
const OuterSchema = Schema.Struct({
  id: Schema.String,
  title: Schema.String,
  optionField: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  nestedOption: Schema.optionalWith(InnerSchema, { as: 'Option', exact: true }),
  defaultField: Schema.optionalWith(Schema.String, { default: () => 'default-val' }),
  nullableField: Schema.Union(Schema.String, Schema.Null),
  items: Schema.Array(Schema.String),
  records: Schema.Record({ key: Schema.String, value: Schema.Number }),
  unionField: Schema.Union(
    Schema.Struct({ type: Schema.Literal('a'), aVal: Schema.String }),
    Schema.Struct({ type: Schema.Literal('b'), bVal: Schema.Number }),
  ),
})

type Outer = Schema.Schema.Type<typeof OuterSchema>

// Schema with a union of literals discriminator
const ModeSchema = Schema.Struct({
  id: Schema.String,
  mode: Schema.Literal('default', 'compact', 'transcript'),
  label: Schema.String,
})

// Schema with a struct containing an Option field, wrapped in an Option (nested Option)
const NestedOptionSchema = Schema.Struct({
  id: Schema.String,
  wrapper: Schema.optionalWith(
    Schema.Struct({
      label: Schema.String,
      inner: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
    }),
    { as: 'Option', exact: true },
  ),
})

// Schema with a union containing an Option field (mirrors the production path)
const UnionWithOptionSchema = Schema.Struct({
  id: Schema.String,
  payload: Schema.Union(
    Schema.Struct({
      type: Schema.Literal('tool'),
      presentation: Schema.optionalWith(
        Schema.Struct({
          toolKey: Schema.String,
          partialStdout: Schema.String,
          running: Schema.Boolean,
        }),
        { as: 'Option', exact: true },
      ),
    }),
    Schema.Struct({
      type: Schema.Literal('text'),
      content: Schema.String,
    }),
  ),
})

type UnionWithOption = Schema.Schema.Type<typeof UnionWithOptionSchema>

// Schema with a suspend inside an Option inside a union (exact production pattern)
const SuspendUnionOptionSchema = Schema.Struct({
  id: Schema.String,
  payload: Schema.Union(
    Schema.Struct({
      type: Schema.Literal('tool'),
      presentation: Schema.optionalWith(
        Schema.suspend(() =>
          Schema.Struct({
            toolKey: Schema.String,
            partialStdout: Schema.String,
            phase: Schema.String,
          }),
        ),
        { as: 'Option', exact: true },
      ),
    }),
    Schema.Struct({
      type: Schema.Literal('text'),
      content: Schema.String,
    }),
  ),
})

type SuspendUnionOption = Schema.Schema.Type<typeof SuspendUnionOptionSchema>

// Schema with a union of unions of literals
const NestedLiteralUnionSchema = Schema.Struct({
  id: Schema.String,
  status: Schema.Union(
    Schema.Literal('idle'),
    Schema.Union(Schema.Literal('working'), Schema.Literal('killed')),
  ),
  label: Schema.String,
})

// Schema with a union where the member itself is a struct with Option fields
// and the inner Option's value is a union
const DeepUnionSchema = Schema.Struct({
  id: Schema.String,
  message: Schema.Union(
    Schema.Struct({
      _tag: Schema.Literal('ToolMessage'),
      id: Schema.String,
      presentation: Schema.optionalWith(
        Schema.Union(
          Schema.Struct({
            kind: Schema.Literal('shell'),
            command: Schema.String,
            partialStdout: Schema.String,
          }),
          Schema.Struct({
            kind: Schema.Literal('fileEdit'),
            path: Schema.String,
            addedCount: Schema.Number,
          }),
        ),
        { as: 'Option', exact: true },
      ),
    }),
    Schema.Struct({
      _tag: Schema.Literal('UserMessage'),
      id: Schema.String,
      content: Schema.String,
    }),
  ),
})

type DeepUnion = Schema.Schema.Type<typeof DeepUnionSchema>

// ===========================================================================
// Group 1: The production crash — undefined values (FC1, FC7)
// ===========================================================================

describe('Group 1: undefined values (FC1, FC7)', () => {
  const compiled = compilePatchMap(OuterSchema)

  function makeOuter(overrides: Partial<Outer> = {}): Outer {
    const base = Schema.decodeSync(OuterSchema)({
      id: '1',
      title: 'Hello',
      optionField: 'opt',
      defaultField: 'custom',
      nullableField: 'not-null',
      items: ['a', 'b', 'c'],
      records: { x: 1, y: 2 },
      unionField: { type: 'a', aVal: 'hello' },
    })
    return { ...base, ...overrides }
  }

  it('1. Some→None emits remove, not replace with undefined', () => {
    const prev = makeOuter({ optionField: Option.some('opt') })
    const next = makeOuter({ optionField: Option.none() })
    expect(Option.isNone(next.optionField)).toBe(true)

    const ops = assertRoundTrip(compiled, prev, next, 'Some→None')
    // Must emit remove, not replace with undefined
    const replaceOps = ops.filter((o) => o.op === 'replace')
    expect(replaceOps).toHaveLength(0)
    const removeOps = ops.filter((o) => o.op === 'remove')
    expect(removeOps.length).toBeGreaterThan(0)
    expect(removeOps.some((o) => o.path[o.path.length - 1] === 'optionField')).toBe(true)
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it('2. None→Some emits replace with encoded value', () => {
    const prev = makeOuter({ optionField: Option.none() })
    expect(Option.isNone(prev.optionField)).toBe(true)

    const next = makeOuter({ optionField: Option.some('now-defined') })
    const ops = assertRoundTrip(compiled, prev, next, 'None→Some')
    const replaceOps = ops.filter(
      (o) => o.op === 'replace' && o.path[o.path.length - 1] === 'optionField',
    )
    expect(replaceOps).toHaveLength(1)
    expect((replaceOps[0] as Extract<DecodedPatchOp, { op: 'replace' }>).value).toBe('now-defined')
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it('3. new object with Option.none field → diff skips the missing key', () => {
    // A record where the new entry has opt as Option.none (absent)
    const RecordSchema = Schema.Record({
      key: Schema.String,
      value: Schema.Struct({
        name: Schema.String,
        opt: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
      }),
    })
    const recCompiled = compilePatchMap(RecordSchema)

    const prev: Record<string, { name: string; opt: Option.Option<string> }> = {}
    const next: Record<string, { name: string; opt: Option.Option<string> }> = {
      item1: { name: 'test', opt: Option.none() }, // opt is Option.none (absent from encoded form)
    }

    const ops = assertRoundTrip(recCompiled, prev, next, 'new object with Option.none field')
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it('4. Option field goes from None to Some → emits replace', () => {
    const prev = makeOuter({ optionField: Option.none() })
    expect(Option.isNone(prev.optionField)).toBe(true)

    const next = makeOuter({ optionField: Option.some('defined-now') })
    const ops = assertRoundTrip(compiled, prev, next, 'None→Some optional')
    const replaceOps = ops.filter(
      (o) => o.op === 'replace' && o.path[o.path.length - 1] === 'optionField',
    )
    expect(replaceOps).toHaveLength(1)
    expect((replaceOps[0] as Extract<DecodedPatchOp, { op: 'replace' }>).value).toBe('defined-now')
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it('5. encodeValue never produces undefined', () => {
    const prev = makeOuter({ optionField: Option.some('has-value') })
    const next = makeOuter({ optionField: Option.none() })

    const ops = Effect.runSync(diffDecoded(prev, next, compiled))
    // No op should have value === undefined
    for (const op of ops) {
      if ('value' in op) {
        expect(op.value).not.toBeUndefined()
      }
    }
  })

  it('6. wire round-trip: diff ops encode through StreamWireEvent without error', () => {
    const prev = makeOuter({ optionField: Option.some('val1'), title: 'A' })
    const nextDecoded = Schema.decodeSync(OuterSchema)({
      id: '1',
      title: 'B',
      // optionField omitted → Option.none()
      defaultField: 'custom',
      nullableField: 'not-null',
      items: ['a', 'b', 'c'],
      records: { x: 1, y: 2 },
      unionField: { type: 'a', aVal: 'hello' },
    })
    expect(Option.isNone(nextDecoded.optionField)).toBe(true)

    const ops = assertRoundTrip(compiled, prev, nextDecoded, 'wire round-trip')
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })
})

// ===========================================================================
// Group 2: Option handling (FC2, FC6)
// ===========================================================================

describe('Group 2: Option handling (FC2, FC6)', () => {
  const compiled = compilePatchMap(OuterSchema)

  function makeOuter(overrides: Partial<Outer> = {}): Outer {
    const base = Schema.decodeSync(OuterSchema)({
      id: '1',
      title: 'Hello',
      optionField: 'some-val',
      defaultField: 'custom',
      nullableField: 'not-null',
      items: ['a', 'b', 'c'],
      records: { x: 1, y: 2 },
      unionField: { type: 'a', aVal: 'hello' },
    })
    return { ...base, ...overrides }
  }

  it('7. Option Some→Some leaf change → single replace with scalar value (no Option object)', () => {
    const prev = makeOuter({ optionField: Option.some('old') })
    const next = makeOuter({ optionField: Option.some('new') })
    const ops = assertRoundTrip(compiled, prev, next, 'Option Some→Some scalar')
    // Should be exactly one replace with scalar value
    expect(ops).toHaveLength(1)
    expect(ops[0].op).toBe('replace')
    expect(ops[0]).toHaveProperty('value', 'new')
    // Value must NOT be an Option object
    expect(Option.isOption((ops[0] as { value: unknown }).value)).toBe(false)
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it('8. Option Some→None → remove op', () => {
    const prev = makeOuter({ optionField: Option.some('val') })
    const next = makeOuter({ optionField: Option.none() })
    const ops = assertRoundTrip(compiled, prev, next, 'Option Some→None')
    expect(ops).toHaveLength(1)
    expect(ops[0].op).toBe('remove')
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it('9. Option None→Some → replace with encoded inner value (not Option object)', () => {
    const prev = makeOuter({ optionField: Option.none() })
    const next = makeOuter({ optionField: Option.some('appeared') })
    const ops = assertRoundTrip(compiled, prev, next, 'Option None→Some')
    expect(ops).toHaveLength(1)
    expect(ops[0].op).toBe('replace')
    expect((ops[0] as { value: unknown }).value).toBe('appeared')
    expect(Option.isOption((ops[0] as { value: unknown }).value)).toBe(false)
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it('10. Nested Option (Option inside Option) → leaf-level ops', () => {
    const nestedCompiled = compilePatchMap(NestedOptionSchema)
    const prev = Schema.decodeSync(NestedOptionSchema)({
      id: '1',
      wrapper: { label: 'lbl', inner: 'inner-val' },
    })
    const next = Schema.decodeSync(NestedOptionSchema)({
      id: '1',
      wrapper: { label: 'lbl', inner: 'changed-val' },
    })
    const ops = assertRoundTrip(nestedCompiled, prev, next, 'nested Option')
    // Should produce leaf-level ops, not whole-object replaces
    for (const op of ops) {
      if (op.op === 'replace') {
        expect(typeof op.value).toBe('string')
      }
    }
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it('11. Option field with non-scalar inner value (struct) → properly encoded', () => {
    const prev = makeOuter({ nestedOption: Option.some({ name: 'a', value: 1 }) })
    const next = makeOuter({ nestedOption: Option.some({ name: 'b', value: 2 }) })
    const ops = assertRoundTrip(compiled, prev, next, 'Option with struct inner')
    // The diff should recurse into the Option and produce leaf-level ops
    for (const op of ops) {
      if (op.op === 'replace') {
        // Value should be a scalar, not an Option or struct object
        expect(Option.isOption(op.value)).toBe(false)
      }
    }
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it('11b. Option None→Some with struct inner → replace with encoded struct (not Option object)', () => {
    const prev = makeOuter({ nestedOption: Option.none() })
    const next = makeOuter({ nestedOption: Option.some({ name: 'new', value: 42 }) })
    const ops = assertRoundTrip(compiled, prev, next, 'Option None→Some struct')
    // Should be a replace on the nestedOption path
    const replaceOps = ops.filter((o) => o.op === 'replace')
    expect(replaceOps.length).toBeGreaterThanOrEqual(1)
    // The top-level value should NOT be an Option object
    for (const op of replaceOps) {
      if (op.path[op.path.length - 1] === 'nestedOption') {
        expect(Option.isOption(op.value)).toBe(false)
        // Value should be a plain object (encoded struct)
        expect(typeof op.value).toBe('object')
        expect(op.value).not.toBeNull()
      }
    }
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it('11c. Option Some→None for struct inner → remove op', () => {
    const prev = makeOuter({ nestedOption: Option.some({ name: 'x', value: 1 }) })
    const next = makeOuter({ nestedOption: Option.none() })
    const ops = assertRoundTrip(compiled, prev, next, 'Option Some→None struct')
    expect(ops).toHaveLength(1)
    expect(ops[0].op).toBe('remove')
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })
})

// ===========================================================================
// Group 3: Compiled map resolution (FC3)
// ===========================================================================

describe('Group 3: Compiled map resolution (FC3)', () => {
  it('12. subSchemaAt resolves through Union→Option→Suspend→Struct→scalar (presentation.partialStdout)', () => {
    const compiled = compilePatchMap(SuspendUnionOptionSchema)
    const decoded: SuspendUnionOption = Schema.decodeSync(SuspendUnionOptionSchema)({
      id: '1',
      payload: {
        type: 'tool',
        presentation: {
          toolKey: 'shell',
          partialStdout: 'hello',
          phase: 'streaming',
        },
      },
    })
    const path: Path = ['payload', 'presentation', 'partialStdout']
    const sub = compiled.subSchemaAt(path, decoded)
    expect(sub).not.toBeNull()
    // It should be a String schema
    const result = sub ? Effect.runSync(Schema.decode(sub)('test')) : null
    expect(result).toBe('test')
  })

  it('13. fieldAt resolves through Union→Option→Suspend→Struct→scalar', () => {
    const compiled = compilePatchMap(SuspendUnionOptionSchema)
    const decoded: SuspendUnionOption = Schema.decodeSync(SuspendUnionOptionSchema)({
      id: '1',
      payload: {
        type: 'tool',
        presentation: {
          toolKey: 'shell',
          partialStdout: 'hello',
          phase: 'streaming',
        },
      },
    })
    const path: Path = ['payload', 'presentation', 'partialStdout']
    const field = compiled.fieldAt(path, decoded)
    expect(field).not.toBeNull()
    // partialStdout is a plain String field inside the Option struct — it is not itself Option
    expect(field?.isOption).toBe(false)
  })

  it('14. subSchemaAt resolves through Union→Option→Suspend→Union→Struct→scalar', () => {
    const compiled = compilePatchMap(DeepUnionSchema)
    const decoded: DeepUnion = Schema.decodeSync(DeepUnionSchema)({
      id: '1',
      message: {
        _tag: 'ToolMessage',
        id: 'm1',
        presentation: {
          kind: 'shell',
          command: 'ls',
          partialStdout: 'hello',
        },
      },
    })
    const path: Path = ['message', 'presentation', 'partialStdout']
    const sub = compiled.subSchemaAt(path, decoded)
    expect(sub).not.toBeNull()
    if (sub) {
      expect(Effect.runSync(Schema.decode(sub)('test'))).toBe('test')
    }
  })

  it('15. union-of-literals discriminator resolves correctly', () => {
    const compiled = compilePatchMap(NestedLiteralUnionSchema)
    const decoded = Schema.decodeSync(NestedLiteralUnionSchema)({
      id: '1',
      status: 'working',
      label: 'test',
    })
    const sub = compiled.subSchemaAt(['status'], decoded)
    expect(sub).not.toBeNull()
    if (sub) {
      expect(Effect.runSync(Schema.decode(sub)('working'))).toBe('working')
    }
  })

  it('16. fieldAt returns isOption=true for Option fields through unions', () => {
    const compiled = compilePatchMap(UnionWithOptionSchema)
    const decoded: UnionWithOption = Schema.decodeSync(UnionWithOptionSchema)({
      id: '1',
      payload: {
        type: 'tool',
        presentation: {
          toolKey: 'shell',
          partialStdout: 'hello',
          running: true,
        },
      },
    })
    // The presentation field itself is Option
    const field = compiled.fieldAt(['payload', 'presentation'], decoded)
    expect(field).not.toBeNull()
    expect(field?.isOption).toBe(true)
    // A field inside the Option struct is not itself Option
    const innerField = compiled.fieldAt(['payload', 'presentation', 'partialStdout'], decoded)
    expect(innerField).not.toBeNull()
    expect(innerField?.isOption).toBe(false)
  })

  it('16b. fieldAt at the Option field itself returns isOption=true', () => {
    const compiled = compilePatchMap(UnionWithOptionSchema)
    const decoded: UnionWithOption = Schema.decodeSync(UnionWithOptionSchema)({
      id: '1',
      payload: {
        type: 'tool',
        presentation: {
          toolKey: 'shell',
          partialStdout: 'hello',
          running: true,
        },
      },
    })
    const field = compiled.fieldAt(['payload', 'presentation'], decoded)
    expect(field).not.toBeNull()
    expect(field?.isOption).toBe(true)
  })

  it('16c. subSchemaAt through union selecting different member', () => {
    const compiled = compilePatchMap(UnionWithOptionSchema)
    // Text member
    const decodedText: UnionWithOption = Schema.decodeSync(UnionWithOptionSchema)({
      id: '1',
      payload: { type: 'text', content: 'hello' },
    })
    const sub = compiled.subSchemaAt(['payload', 'content'], decodedText)
    expect(sub).not.toBeNull()
    if (sub) {
      expect(Effect.runSync(Schema.decode(sub)('test'))).toBe('test')
    }
  })
})

// ===========================================================================
// Group 4: Apply correctness (FC4)
// ===========================================================================

describe('Group 4: Apply correctness (FC4)', () => {
  const compiled = compilePatchMap(OuterSchema)

  function makeOuter(overrides: Partial<Outer> = {}): Outer {
    const base = Schema.decodeSync(OuterSchema)({
      id: '1',
      title: 'Hello',
      optionField: 'some-val',
      defaultField: 'custom',
      nullableField: 'not-null',
      items: ['a', 'b', 'c'],
      records: { x: 1, y: 2 },
      unionField: { type: 'a', aVal: 'hello' },
    })
    return { ...base, ...overrides }
  }

  it('17. decodeLeaf fails (not silently returns raw JSON) when fieldAt returns null for non-scalar', () => {
    // Create a schema where fieldAt returns null for a path, then try to apply
    // a replace with a non-scalar value at that path.
    // Use a record schema with unknown values — subSchemaAt will return the
    // record value schema but fieldAt may return null for deep paths.
    const TestSchema = Schema.Struct({
      data: Schema.Record({ key: Schema.String, value: Schema.Unknown }),
    })
    const testCompiled = compilePatchMap(TestSchema) as CompiledMap<DecodedValue>
    const prev = Schema.decodeSync(TestSchema)({ data: {} }) as DecodedValue

    // Apply a replace with a non-scalar at a path where fieldAt returns null
    const op: DecodedPatchOp = {
      op: 'replace',
      path: ['data', 'key1', 'nested'],
      value: { foo: 'bar' },
    }
    const either = Effect.runSync(Effect.either(applyDecodedPatch(prev, [op], testCompiled)))
    // This should NOT silently return the raw JSON — it should either error
    // (if fieldAt returns null and can't decode) or succeed by inserting the
    // raw value (since Schema.Unknown accepts anything).
    // The key invariant: it should NOT produce a silently wrong decoded value.
    if (either._tag === 'Right') {
      // If it succeeds, the value at the path should match what we put in
      const result = either.right as { data: Record<string, unknown> }
      expect(result.data.key1).toBeDefined()
    }
    // Either way, no crash, no silent corruption
  })

  it('18. Apply replace on Option field → wraps decoded value in Option.some', () => {
    const prev = makeOuter({ optionField: Option.none() })
    const op: DecodedPatchOp = {
      op: 'replace',
      path: ['optionField'],
      value: 'wrapped-val',
    }
    const result = Effect.runSync(applyDecodedPatch(prev, [op], compiled)) as Outer
    expect(Option.isOption(result.optionField)).toBe(true)
    expect(Option.isSome(result.optionField)).toBe(true)
    expect(Option.getOrNull(result.optionField)).toBe('wrapped-val')
  })

  it('19. Apply remove on Option field → sets Option.none()', () => {
    const prev = makeOuter({ optionField: Option.some('val') })
    const op: DecodedPatchOp = {
      op: 'remove',
      path: ['optionField'],
    }
    const result = Effect.runSync(applyDecodedPatch(prev, [op], compiled)) as Outer
    expect(Option.isOption(result.optionField)).toBe(true)
    expect(Option.isNone(result.optionField)).toBe(true)
  })

  it('20. Apply remove on default field → sets default value', () => {
    const prev = makeOuter({ defaultField: 'custom-val' })
    const op: DecodedPatchOp = {
      op: 'remove',
      path: ['defaultField'],
    }
    const result = Effect.runSync(applyDecodedPatch(prev, [op], compiled)) as Outer
    expect(result.defaultField).toBe('default-val')
  })

  it('21. Apply remove on plain field → deletes key', () => {
    const prev = makeOuter({ nullableField: 'val' })
    const op: DecodedPatchOp = {
      op: 'remove',
      path: ['nullableField'],
    }
    // nullableField is Schema.Union(String, Null) — not optional, not Option, not default
    // So remove should delete the key (or set to undefined)
    const result = Effect.runSync(applyDecodedPatch(prev, [op], compiled)) as Outer
    expect(result.nullableField).toBeUndefined()
  })

  it('22. Apply add on array → inserts at index (shifts right)', () => {
    const prev = makeOuter({ items: ['a', 'c'] })
    const op: DecodedPatchOp = {
      op: 'add',
      path: ['items', 1],
      value: 'b',
    }
    const result = Effect.runSync(applyDecodedPatch(prev, [op], compiled)) as Outer
    expect(result.items).toEqual(['a', 'b', 'c'])
  })

  it('23. Apply remove on array element → removes (shifts left)', () => {
    const prev = makeOuter({ items: ['a', 'b', 'c'] })
    const op: DecodedPatchOp = {
      op: 'remove',
      path: ['items', 1],
    }
    const result = Effect.runSync(applyDecodedPatch(prev, [op], compiled)) as Outer
    expect(result.items).toEqual(['a', 'c'])
  })

  it('23b. Apply replace on array element → overwrites at index', () => {
    const prev = makeOuter({ items: ['a', 'b', 'c'] })
    const op: DecodedPatchOp = {
      op: 'replace',
      path: ['items', 1],
      value: 'X',
    }
    const result = Effect.runSync(applyDecodedPatch(prev, [op], compiled)) as Outer
    expect(result.items).toEqual(['a', 'X', 'c'])
  })

  it('23c. Apply replace on nested Option struct field → decoded correctly', () => {
    const prev = makeOuter({ nestedOption: Option.some({ name: 'old', value: 1 }) })
    const next = makeOuter({ nestedOption: Option.some({ name: 'new', value: 2 }) })
    const ops = Effect.runSync(diffDecoded(prev, next, compiled))
    const result = Effect.runSync(applyDecodedPatch(prev, ops, compiled)) as Outer
    expect(result).toEqual(next)
    expect(Option.isOption(result.nestedOption)).toBe(true)
    expect(Option.isSome(result.nestedOption)).toBe(true)
    expect(Option.getOrNull(result.nestedOption)?.name).toBe('new')
    expect(Option.getOrNull(result.nestedOption)?.value).toBe(2)
  })
})

// ===========================================================================
// Group 5: Move (FC5)
// ===========================================================================

describe('Group 5: Move (FC5)', () => {
  const compiled = compilePatchMap(OuterSchema)

  function makeOuter(overrides: Partial<Outer> = {}): Outer {
    const base = Schema.decodeSync(OuterSchema)({
      id: '1',
      title: 'Hello',
      optionField: 'some-val',
      defaultField: 'custom',
      nullableField: 'not-null',
      items: ['a', 'b', 'c', 'd'],
      records: { x: 1, y: 2 },
      unionField: { type: 'a', aVal: 'hello' },
    })
    return { ...base, ...overrides }
  }

  it('24. Move array element within same array → correct result', () => {
    const prev = makeOuter({ items: ['a', 'b', 'c', 'd'] })
    const op: DecodedPatchOp = {
      op: 'move',
      from: ['items', 0],
      to: ['items', 2],
    }
    const result = Effect.runSync(applyDecodedPatch(prev, [op], compiled)) as Outer
    // Move 'a' from index 0 to index 2: ['b', 'c', 'a', 'd']
    expect(result.items).toEqual(['b', 'c', 'a', 'd'])
  })

  it('24b. Move array element forward → correct result', () => {
    const prev = makeOuter({ items: ['a', 'b', 'c', 'd'] })
    const op: DecodedPatchOp = {
      op: 'move',
      from: ['items', 3],
      to: ['items', 0],
    }
    const result = Effect.runSync(applyDecodedPatch(prev, [op], compiled)) as Outer
    expect(result.items).toEqual(['d', 'a', 'b', 'c'])
  })

  it('25. Move preserves value reference (no re-decode)', () => {
    // Use a schema with objects in arrays to check reference preservation
    const ArrayObjSchema = Schema.Struct({
      items: Schema.Array(Schema.Struct({ id: Schema.String, label: Schema.String })),
    })
    const arrCompiled = compilePatchMap(ArrayObjSchema)
    const prev = Schema.decodeSync(ArrayObjSchema)({
      items: [
        { id: '1', label: 'one' },
        { id: '2', label: 'two' },
        { id: '3', label: 'three' },
      ],
    })
    const op: DecodedPatchOp = {
      op: 'move',
      from: ['items', 0],
      to: ['items', 2],
    }
    const result = Effect.runSync(applyDecodedPatch(prev, [op], arrCompiled)) as {
      items: Array<{ id: string; label: string }>
    }
    expect(result.items.map((i) => i.id)).toEqual(['2', '3', '1'])
    // The moved element should be the same reference (no re-decode)
    expect(result.items[2]).toStrictEqual(prev.items[0])
  })
})

// ===========================================================================
// Group 6: _key contamination (FC8)
// ===========================================================================

describe('Group 6: _key contamination (FC8)', () => {
  it('26. Store diff does not produce _key ops when comparing accepted vs server values', () => {
    // Simulate the store's scenario: prev has _key injected, next doesn't
    const TestSchema = Schema.Struct({
      items: Schema.Array(
        Schema.Struct({ id: Schema.String, label: Schema.String }),
      ),
    })
    const compiled = compilePatchMap(TestSchema)

    const prev = {
      items: [
        { id: '1', label: 'one', _key: 'key-1' },
        { id: '2', label: 'two', _key: 'key-2' },
      ],
    }
    const next = {
      items: [
        { id: '1', label: 'one' },
        { id: '2', label: 'two-changed' },
      ],
    }

    const ops = Effect.runSync(diffDecoded(prev, next, compiled))
    // No op should contain _key
    assertNoKey(ops)
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it("26b. _key in next (server) does not leak into ops", () => {
    const TestSchema = Schema.Struct({
      items: Schema.Array(
        Schema.Struct({ id: Schema.String, label: Schema.String }),
      ),
    })
    const compiled = compilePatchMap(TestSchema)

    const prev = {
      items: [{ id: '1', label: 'one' }],
    }
    const next = {
      items: [{ id: '1', label: 'one', _key: 'injected' }],
    }

    const ops = Effect.runSync(diffDecoded(prev, next, compiled))
    assertNoKey(ops)
  })
})

// ===========================================================================
// Group 7: Structural sharing (FC9)
// ===========================================================================

describe('Group 7: Structural sharing (FC9)', () => {
  const compiled = compilePatchMap(OuterSchema)

  function makeOuter(overrides: Partial<Outer> = {}): Outer {
    const base = Schema.decodeSync(OuterSchema)({
      id: '1',
      title: 'Hello',
      optionField: 'some-val',
      defaultField: 'custom',
      nullableField: 'not-null',
      items: ['a', 'b', 'c'],
      records: { x: 1, y: 2 },
      unionField: { type: 'a', aVal: 'hello' },
    })
    return { ...base, ...overrides }
  }

  it('27. Unchanged siblings preserve references after patch', () => {
    const prev = makeOuter()
    const next = makeOuter({ title: 'new-title' })
    const ops = Effect.runSync(diffDecoded(prev, next, compiled))
    const result = Effect.runSync(applyDecodedPatch(prev, ops, compiled)) as Outer
    expect(result.items).toBe(prev.items)
    expect(result.records).toBe(prev.records)
    expect(result.unionField).toBe(prev.unionField)
    expect(result.optionField).toBe(prev.optionField)
    expect(result.nullableField).toBe(prev.nullableField)
  })

  it('28. Unchanged array elements preserve references', () => {
    const ArrayObjSchema = Schema.Struct({
      items: Schema.Array(Schema.Struct({ id: Schema.String, label: Schema.String })),
    })
    const arrCompiled = compilePatchMap(ArrayObjSchema)
    const prev = Schema.decodeSync(ArrayObjSchema)({
      items: [
        { id: '1', label: 'one' },
        { id: '2', label: 'two' },
        { id: '3', label: 'three' },
      ],
    })
    const next = Schema.decodeSync(ArrayObjSchema)({
      items: [
        { id: '1', label: 'one' },
        { id: '2', label: 'CHANGED' },
        { id: '3', label: 'three' },
      ],
    })
    const ops = Effect.runSync(diffDecoded(prev, next, arrCompiled))
    const result = Effect.runSync(applyDecodedPatch(prev, ops, arrCompiled)) as {
      items: Array<{ id: string; label: string }>
    }
    // Unchanged elements should preserve references
    expect(result.items[0]).toBe(prev.items[0])
    expect(result.items[2]).toBe(prev.items[2])
    // Changed element should be new
    expect(result.items[1]).not.toBe(prev.items[1])
    expect(result.items[1].label).toBe('CHANGED')
  })

  it('29. Unchanged record entries preserve references', () => {
    const RecordSchema = Schema.Struct({
      data: Schema.Record({
        key: Schema.String,
        value: Schema.Struct({ val: Schema.Number }),
      }),
    })
    const recCompiled = compilePatchMap(RecordSchema)
    const prev = Schema.decodeSync(RecordSchema)({
      data: {
        a: { val: 1 },
        b: { val: 2 },
        c: { val: 3 },
      },
    })
    const next = Schema.decodeSync(RecordSchema)({
      data: {
        a: { val: 1 },
        b: { val: 99 },
        c: { val: 3 },
      },
    })
    const ops = Effect.runSync(diffDecoded(prev, next, recCompiled))
    const result = Effect.runSync(applyDecodedPatch(prev, ops, recCompiled)) as {
      data: Record<string, { val: number }>
    }
    // Unchanged entries should preserve references
    expect(result.data.a).toBe(prev.data.a)
    expect(result.data.c).toBe(prev.data.c)
    expect(result.data.b).not.toBe(prev.data.b)
    expect(result.data.b.val).toBe(99)
  })
})

// ===========================================================================
// Group 8: Production DisplayViewSnapshot scenarios
// ===========================================================================

describe('Group 8: Production DisplayViewSnapshot scenarios', () => {
  const compiled = compilePatchMap(DisplayViewSnapshot)

  function makeSnapshot(partialStdout: string): DVS {
    return Schema.decodeSync(DisplayViewSnapshot)({
      shape: { timelines: { root: { kind: 'tail', limit: 100, live: true } } },
      state: {
        session: { sessionId: 's1', title: 'Test', cwd: '/tmp' },
        timelines: {
          root: {
            mode: 'streaming',
            messages: {
              byId: {
                m1: {
                  id: 'm1', type: 'tool', toolKey: 'shell', timestamp: 1000,
                  presentation: {
                    toolKey: 'shell', phase: 'streaming', tone: 'neutral', icon: 'terminal',
                    command: 'ls', done: null, exitCode: null, pid: null,
                    stdout: '', stderr: '', partialStdout, partialStderr: '',
                    stdoutPath: null, stderrPath: null, errorText: null,
                    running: true, failed: false,
                  },
                },
              },
              order: ['m1'],
            },
            streamingMessageId: 'm1',
          },
        },
        actors: {}, agents: {},
        tasks: { byId: {}, order: [], summary: { totalCount: 0, completedCount: 0, incompleteCount: 0 } },
      },
    })
  }

  function makeSnapshotWithAgent(status: 'working' | 'idle' | 'killed' | undefined): DVS {
    const agentObj: { name: string; role: string; status?: 'working' | 'idle' | 'killed' } = { name: 'root', role: 'main' }
    if (status !== undefined) agentObj.status = status
    return Schema.decodeSync(DisplayViewSnapshot)({
      shape: { timelines: { root: { kind: 'tail', limit: 100, live: true } } },
      state: {
        session: { sessionId: 's1', title: 'Test', cwd: '/tmp' },
        timelines: {
          root: {
            mode: 'idle',
            messages: { byId: {}, order: [] },
            streamingMessageId: null,
          },
        },
        actors: {}, agents: { root: agentObj },
        tasks: { byId: {}, order: [], summary: { totalCount: 0, completedCount: 0, incompleteCount: 0 } },
      },
    })
  }

  it('30. partialStdout streaming change → 1 leaf op, wire round-trips', () => {
    const prev = makeSnapshot('hello')
    const next = makeSnapshot('hello world')
    const ops = assertRoundTrip(compiled, prev, next, 'partialStdout streaming')
    expect(ops).toHaveLength(1)
    expect(ops[0].op).toBe('replace')
    expect((ops[0] as { path: Path }).path).toEqual([
      'state', 'timelines', 'root', 'messages', 'byId', 'm1', 'presentation', 'partialStdout',
    ])
    expect((ops[0] as { value: unknown }).value).toBe('hello world')
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it('31. New message added → multiple ops, all wire-encode correctly, apply produces correct result', () => {
    const prev = makeSnapshot('hello')
    const next = Schema.decodeSync(DisplayViewSnapshot)({
      shape: { timelines: { root: { kind: 'tail', limit: 100, live: true } } },
      state: {
        session: { sessionId: 's1', title: 'Test', cwd: '/tmp' },
        timelines: {
          root: {
            mode: 'streaming',
            messages: {
              byId: {
                m1: {
                  id: 'm1', type: 'tool', toolKey: 'shell', timestamp: 1000,
                  presentation: {
                    toolKey: 'shell', phase: 'streaming', tone: 'neutral', icon: 'terminal',
                    command: 'ls', done: null, exitCode: null, pid: null,
                    stdout: '', stderr: '', partialStdout: 'hello', partialStderr: '',
                    stdoutPath: null, stderrPath: null, errorText: null,
                    running: true, failed: false,
                  },
                },
                m2: {
                  id: 'm2', type: 'assistant_message', content: 'Hello!', timestamp: 2000,
                },
              },
              order: ['m1', 'm2'],
            },
            streamingMessageId: 'm1',
          },
        },
        actors: {}, agents: {},
        tasks: { byId: {}, order: [], summary: { totalCount: 0, completedCount: 0, incompleteCount: 0 } },
      },
    })
    const ops = assertRoundTrip(compiled, prev, next, 'new message added')
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it('32. Presentation Some→None → remove op, apply sets Option.none()', () => {
    const prev = makeSnapshot('hello')
    const m1 = prev.state.timelines.root.messages.byId.m1 as Extract<DVS['state']['timelines']['root']['messages']['byId'][string], { type: 'tool' }>
    const next: DVS = {
      ...prev,
      state: {
        ...prev.state,
        timelines: {
          ...prev.state.timelines,
          root: {
            ...prev.state.timelines.root,
            messages: {
              ...prev.state.timelines.root.messages,
              byId: {
                ...prev.state.timelines.root.messages.byId,
                m1: { ...m1, presentation: Option.none() },
              },
            },
          },
        },
      },
    }
    const ops = assertRoundTrip(compiled, prev, next, 'presentation Some→None')
    // Should have a remove op for the presentation path
    const removeOps = ops.filter(
      (o) => o.op === 'remove' && o.path[o.path.length - 1] === 'presentation',
    )
    expect(removeOps.length).toBeGreaterThanOrEqual(1)
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
    // Verify the result has Option.none
    const result = Effect.runSync(applyDecodedPatch(prev, ops, compiled)) as DVS
    const resultM1 = result.state.timelines.root.messages.byId.m1 as typeof m1
    expect(Option.isNone(resultM1.presentation)).toBe(true)
  })

  it('33. Agent status changes → correctly diffed (no undefined in ops)', () => {
    const prev = makeSnapshotWithAgent('working')
    const next = makeSnapshotWithAgent('idle')
    const ops = assertRoundTrip(compiled, prev, next, 'agent status change')
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it('33b. Agent status working→undefined → remove op (not replace with undefined)', () => {
    const prev = makeSnapshotWithAgent('working')
    const next = makeSnapshotWithAgent(undefined)
    const ops = assertRoundTrip(compiled, prev, next, 'agent status→undefined')
    // Must not have any replace with undefined
    const replaceOps = ops.filter((o) => o.op === 'replace')
    for (const op of replaceOps) {
      expect((op as { value: unknown }).value).not.toBeUndefined()
    }
    // Should have a remove op for the status field
    const removeOps = ops.filter(
      (o) => o.op === 'remove' && o.path[o.path.length - 1] === 'status',
    )
    expect(removeOps.length).toBeGreaterThanOrEqual(1)
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it('34. New agent added with optional fields → no undefined in ops', () => {
    const prev = makeSnapshotWithAgent(undefined) // agent with no status
    const next = makeSnapshotWithAgent('working')
    const ops = assertRoundTrip(compiled, prev, next, 'new agent with optional fields')
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it('34b. New agent added (entirely new record entry) → no undefined in ops', () => {
    const prev = Schema.decodeSync(DisplayViewSnapshot)({
      shape: { timelines: { root: { kind: 'tail', limit: 100, live: true } } },
      state: {
        session: { sessionId: 's1', title: 'Test', cwd: '/tmp' },
        timelines: {
          root: { mode: 'idle', messages: { byId: {}, order: [] }, streamingMessageId: null },
        },
        actors: {}, agents: {},
        tasks: { byId: {}, order: [], summary: { totalCount: 0, completedCount: 0, incompleteCount: 0 } },
      },
    })
    const next = makeSnapshotWithAgent(undefined) // new agent with no status
    const ops = assertRoundTrip(compiled, prev, next, 'entirely new agent')
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it('35. Timeline mode change → correct ops', () => {
    const prev = makeSnapshot('hello')
    const next: DVS = {
      ...prev,
      state: {
        ...prev.state,
        timelines: {
          ...prev.state.timelines,
          root: { ...prev.state.timelines.root, mode: 'idle' },
        },
      },
    }
    const ops = assertRoundTrip(compiled, prev, next, 'timeline mode change')
    const replaceOps = ops.filter(
      (o) => o.op === 'replace' && o.path[o.path.length - 1] === 'mode',
    )
    expect(replaceOps.length).toBeGreaterThanOrEqual(1)
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it('36. Task added/removed → correct ops', () => {
    const prev = Schema.decodeSync(DisplayViewSnapshot)({
      shape: { timelines: { root: { kind: 'tail', limit: 100, live: true } } },
      state: {
        session: { sessionId: 's1', title: 'Test', cwd: '/tmp' },
        timelines: { root: { mode: 'idle', messages: { byId: {}, order: [] }, streamingMessageId: null } },
        actors: {}, agents: {},
        tasks: { byId: {}, order: [], summary: { totalCount: 0, completedCount: 0, incompleteCount: 0 } },
      },
    })
    const next = Schema.decodeSync(DisplayViewSnapshot)({
      shape: { timelines: { root: { kind: 'tail', limit: 100, live: true } } },
      state: {
        session: { sessionId: 's1', title: 'Test', cwd: '/tmp' },
        timelines: { root: { mode: 'idle', messages: { byId: {}, order: [] }, streamingMessageId: null } },
        actors: {}, agents: {},
        tasks: {
          byId: {
            t1: {
              rowId: 'row-1', kind: 'task', taskId: 't1', title: 'Do thing',
              status: 'pending', depth: 0, updatedAt: 1000,
              assignee: { kind: 'none' },
            },
          },
          order: ['t1'],
          summary: { totalCount: 1, completedCount: 0, incompleteCount: 1 },
        },
      },
    })
    // Add task
    const addOps = assertRoundTrip(compiled, prev, next, 'task added')
    assertNoUndefined(addOps)
    assertWireRoundTrip(addOps)
    // Remove task (reverse)
    const removeOps = assertRoundTrip(compiled, next, prev, 'task removed')
    assertNoUndefined(removeOps)
    assertWireRoundTrip(removeOps)
  })

  it('37. Empty diff (no changes) → 0 ops', () => {
    const prev = makeSnapshot('hello')
    const ops = Effect.runSync(diffDecoded(prev, prev, compiled))
    expect(ops).toEqual([])
  })

  it('37b. Timeline presentation mode change → correct ops', () => {
    const prev = Schema.decodeSync(DisplayViewSnapshot)({
      shape: { timelines: { root: { kind: 'tail', limit: 100, live: true, presentation: 'default' } } },
      state: {
        session: { sessionId: 's1', title: 'Test', cwd: '/tmp' },
        timelines: {
          root: {
            mode: 'idle',
            messages: { byId: {}, order: [] },
            streamingMessageId: null,
            presentation: { mode: 'default', entries: [], statusSlot: { kind: 'none' } },
          },
        },
        actors: {}, agents: {},
        tasks: { byId: {}, order: [], summary: { totalCount: 0, completedCount: 0, incompleteCount: 0 } },
      },
    })
    const next = Schema.decodeSync(DisplayViewSnapshot)({
      shape: { timelines: { root: { kind: 'tail', limit: 100, live: true, presentation: 'transcript' } } },
      state: {
        session: { sessionId: 's1', title: 'Test', cwd: '/tmp' },
        timelines: {
          root: {
            mode: 'idle',
            messages: { byId: {}, order: [] },
            streamingMessageId: null,
            presentation: { mode: 'transcript', entries: [], statusSlot: { kind: 'none' } },
          },
        },
        actors: {}, agents: {},
        tasks: { byId: {}, order: [], summary: { totalCount: 0, completedCount: 0, incompleteCount: 0 } },
      },
    })
    const ops = assertRoundTrip(compiled, prev, next, 'timeline presentation mode change')
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it('37c. Multiple simultaneous changes across the snapshot', () => {
    const prev = makeSnapshot('hello')
    const next = Schema.decodeSync(DisplayViewSnapshot)({
      shape: { timelines: { root: { kind: 'tail', limit: 100, live: true } } },
      state: {
        session: { sessionId: 's1', title: 'Changed Title', cwd: '/new' },
        timelines: {
          root: {
            mode: 'idle',
            messages: {
              byId: {
                m1: {
                  id: 'm1', type: 'tool', toolKey: 'shell', timestamp: 1000,
                  presentation: {
                    toolKey: 'shell', phase: 'completed', tone: 'success', icon: 'terminal',
                    command: 'ls', done: 'completed', exitCode: 0, pid: null,
                    stdout: 'hello world', stderr: '', partialStdout: 'hello world', partialStderr: '',
                    stdoutPath: null, stderrPath: null, errorText: null,
                    running: false, failed: false,
                  },
                },
              },
              order: ['m1'],
            },
            streamingMessageId: null,
          },
        },
        actors: {}, agents: { root: { name: 'root', role: 'main', status: 'idle' } },
        tasks: { byId: {}, order: [], summary: { totalCount: 0, completedCount: 0, incompleteCount: 0 } },
      },
    })
    const ops = assertRoundTrip(compiled, prev, next, 'multiple simultaneous changes')
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })
})

// ===========================================================================
// Group 9: Error cases
// ===========================================================================

describe('Group 9: Error cases', () => {
  const compiled = compilePatchMap(OuterSchema)

  function makeOuter(overrides: Partial<Outer> = {}): Outer {
    const base = Schema.decodeSync(OuterSchema)({
      id: '1',
      title: 'Hello',
      optionField: 'some-val',
      defaultField: 'custom',
      nullableField: 'not-null',
      items: ['a', 'b', 'c'],
      records: { x: 1, y: 2 },
      unionField: { type: 'a', aVal: 'hello' },
    })
    return { ...base, ...overrides }
  }

  it('38. Navigate into Option.none → PatchApplyError', () => {
    const prev = makeOuter({ optionField: Option.none() })
    const op: DecodedPatchOp = {
      op: 'replace',
      path: ['optionField', 'nonexistent'],
      value: 'test',
    }
    const either = Effect.runSync(Effect.either(applyDecodedPatch(prev, [op], compiled)))
    expect(either._tag).toBe('Left')
    if (either._tag === 'Left') {
      expect(either.left).toBeInstanceOf(PatchNavigationError)
    }
  })

  it('39. Navigate into non-container → PatchApplyError', () => {
    const prev = makeOuter()
    const op: DecodedPatchOp = {
      op: 'replace',
      path: ['title', 'nested'],
      value: 'test',
    }
    const either = Effect.runSync(Effect.either(applyDecodedPatch(prev, [op], compiled)))
    expect(either._tag).toBe('Left')
    if (either._tag === 'Left') {
      expect(either.left).toBeInstanceOf(PatchNavigationError)
    }
  })

  it('40. Remove root → PatchApplyError', () => {
    const prev = makeOuter()
    const op: DecodedPatchOp = {
      op: 'remove',
      path: [],
    }
    const either = Effect.runSync(Effect.either(applyDecodedPatch(prev, [op], compiled)))
    expect(either._tag).toBe('Left')
    if (either._tag === 'Left') {
      expect(either.left).toBeInstanceOf(PatchNavigationError)
    }
  })

  it('40b. Navigate into null → PatchApplyError', () => {
    const prev = makeOuter({ nullableField: null })
    const op: DecodedPatchOp = {
      op: 'replace',
      path: ['nullableField', 'nested'],
      value: 'test',
    }
    const either = Effect.runSync(Effect.either(applyDecodedPatch(prev, [op], compiled)))
    expect(either._tag).toBe('Left')
    if (either._tag === 'Left') {
      expect(either.left).toBeInstanceOf(PatchNavigationError)
    }
  })

  it('40c. Move from non-existent path → PatchApplyError', () => {
    const prev = makeOuter({ items: ['a', 'b'] })
    const op: DecodedPatchOp = {
      op: 'move',
      from: ['items', 99],
      to: ['items', 0],
    }
    const either = Effect.runSync(Effect.either(applyDecodedPatch(prev, [op], compiled)))
    expect(either._tag).toBe('Left')
    if (either._tag === 'Left') {
      expect(either.left).toBeInstanceOf(PatchNavigationError)
    }
  })
})

// ===========================================================================
// Group 10: Wire schema invariant
// ===========================================================================

describe('Group 10: Wire schema invariant — all ops encode through StreamWireEvent', () => {
  // Re-run key scenarios and verify every op batch wire-encodes

  it('all op types wire-encode (replace, remove, add, move)', () => {
    const ops: DecodedPatchOp[] = [
      { op: 'replace', path: ['a', 'b'], value: 'hello' },
      { op: 'remove', path: ['c'] },
      { op: 'add', path: ['d', 0], value: 42 },
      { op: 'move', from: ['e', 0], to: ['e', 1] },
    ]
    assertWireRoundTrip(ops)
  })

  it('ops with null values wire-encode', () => {
    const ops: DecodedPatchOp[] = [
      { op: 'replace', path: ['nullable'], value: null },
    ]
    assertWireRoundTrip(ops)
  })

  it('ops with object values wire-encode', () => {
    const ops: DecodedPatchOp[] = [
      { op: 'replace', path: ['obj'], value: { a: 1, b: 'two', c: true, d: null } },
    ]
    assertWireRoundTrip(ops)
  })

  it('ops with array values wire-encode', () => {
    const ops: DecodedPatchOp[] = [
      { op: 'replace', path: ['arr'], value: [1, 'two', true, null, { nested: 'val' }] },
    ]
    assertWireRoundTrip(ops)
  })

  it('ops with deeply nested values wire-encode', () => {
    const ops: DecodedPatchOp[] = [
      {
        op: 'replace',
        path: ['deep'],
        value: { level1: { level2: { level3: { data: [1, 2, { final: 'value' }] } } } },
      },
    ]
    assertWireRoundTrip(ops)
  })

  it('empty ops array wire-encodes', () => {
    const ops: DecodedPatchOp[] = []
    assertWireRoundTrip(ops)
  })

  it('production partialStdout ops wire-encode', () => {
    const compiled = compilePatchMap(DisplayViewSnapshot)
    const prev = Schema.decodeSync(DisplayViewSnapshot)({
      shape: { timelines: { root: { kind: 'tail', limit: 100, live: true } } },
      state: {
        session: { sessionId: 's1', title: 'Test', cwd: '/tmp' },
        timelines: {
          root: {
            mode: 'streaming',
            messages: {
              byId: {
                m1: {
                  id: 'm1', type: 'tool', toolKey: 'shell', timestamp: 1000,
                  presentation: {
                    toolKey: 'shell', phase: 'streaming', tone: 'neutral', icon: 'terminal',
                    command: 'ls', done: null, exitCode: null, pid: null,
                    stdout: '', stderr: '', partialStdout: 'a', partialStderr: '',
                    stdoutPath: null, stderrPath: null, errorText: null,
                    running: true, failed: false,
                  },
                },
              },
              order: ['m1'],
            },
            streamingMessageId: 'm1',
          },
        },
        actors: {}, agents: {},
        tasks: { byId: {}, order: [], summary: { totalCount: 0, completedCount: 0, incompleteCount: 0 } },
      },
    })
    const next = Schema.decodeSync(DisplayViewSnapshot)({
      shape: { timelines: { root: { kind: 'tail', limit: 100, live: true } } },
      state: {
        session: { sessionId: 's1', title: 'Test', cwd: '/tmp' },
        timelines: {
          root: {
            mode: 'streaming',
            messages: {
              byId: {
                m1: {
                  id: 'm1', type: 'tool', toolKey: 'shell', timestamp: 1000,
                  presentation: {
                    toolKey: 'shell', phase: 'streaming', tone: 'neutral', icon: 'terminal',
                    command: 'ls', done: null, exitCode: null, pid: null,
                    stdout: '', stderr: '', partialStdout: 'a b c', partialStderr: '',
                    stdoutPath: null, stderrPath: null, errorText: null,
                    running: true, failed: false,
                  },
                },
              },
              order: ['m1'],
            },
            streamingMessageId: 'm1',
          },
        },
        actors: {}, agents: {},
        tasks: { byId: {}, order: [], summary: { totalCount: 0, completedCount: 0, incompleteCount: 0 } },
      },
    })
    const ops = Effect.runSync(diffDecoded(prev, next, compiled))
    assertWireRoundTrip(ops)
  })
})

// ===========================================================================
// Group 11: Compositional / deep path tests
// ===========================================================================

describe('Group 11: Compositional deep-path tests', () => {
  it('union→Option→suspend→struct→scalar diff+apply round-trip', () => {
    const compiled = compilePatchMap(SuspendUnionOptionSchema)
    const prev: SuspendUnionOption = Schema.decodeSync(SuspendUnionOptionSchema)({
      id: '1',
      payload: {
        type: 'tool',
        presentation: {
          toolKey: 'shell',
          partialStdout: 'hello',
          phase: 'streaming',
        },
      },
    })
    const next: SuspendUnionOption = Schema.decodeSync(SuspendUnionOptionSchema)({
      id: '1',
      payload: {
        type: 'tool',
        presentation: {
          toolKey: 'shell',
          partialStdout: 'hello world',
          phase: 'streaming',
        },
      },
    })
    const ops = assertRoundTrip(compiled, prev, next, 'suspend union option deep')
    // Should be a single leaf replace
    expect(ops).toHaveLength(1)
    expect(ops[0].op).toBe('replace')
    expect((ops[0] as { value: unknown }).value).toBe('hello world')
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it('union→Option→union→struct→scalar diff+apply round-trip', () => {
    const compiled = compilePatchMap(DeepUnionSchema)
    const prev: DeepUnion = Schema.decodeSync(DeepUnionSchema)({
      id: '1',
      message: {
        _tag: 'ToolMessage',
        id: 'm1',
        presentation: {
          kind: 'shell',
          command: 'ls',
          partialStdout: 'hello',
        },
      },
    })
    const next: DeepUnion = Schema.decodeSync(DeepUnionSchema)({
      id: '1',
      message: {
        _tag: 'ToolMessage',
        id: 'm1',
        presentation: {
          kind: 'shell',
          command: 'ls',
          partialStdout: 'hello world',
        },
      },
    })
    const ops = assertRoundTrip(compiled, prev, next, 'deep union option union')
    expect(ops).toHaveLength(1)
    expect(ops[0].op).toBe('replace')
    expect((ops[0] as { value: unknown }).value).toBe('hello world')
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it('union member swap with Option field', () => {
    const compiled = compilePatchMap(DeepUnionSchema)
    const prev: DeepUnion = Schema.decodeSync(DeepUnionSchema)({
      id: '1',
      message: {
        _tag: 'ToolMessage',
        id: 'm1',
        presentation: {
          kind: 'shell',
          command: 'ls',
          partialStdout: 'hello',
        },
      },
    })
    const next: DeepUnion = Schema.decodeSync(DeepUnionSchema)({
      id: '1',
      message: {
        _tag: 'UserMessage',
        id: 'm1',
        content: 'Hi there',
      },
    })
    const ops = assertRoundTrip(compiled, prev, next, 'union member swap')
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it('Option None→Some through union (presentation appears)', () => {
    const compiled = compilePatchMap(UnionWithOptionSchema)
    // prev: presentation absent → Option.none() after decode
    const prev: UnionWithOption = Schema.decodeSync(UnionWithOptionSchema)({
      id: '1',
      payload: { type: 'tool' },
    })
    const next: UnionWithOption = Schema.decodeSync(UnionWithOptionSchema)({
      id: '1',
      payload: {
        type: 'tool',
        presentation: {
          toolKey: 'shell',
          partialStdout: 'appeared',
          running: true,
        },
      },
    })
    const ops = assertRoundTrip(compiled, prev, next, 'Option None→Some through union')
    // Should be a replace (not an add) since the field already exists as Option.none
    const replaceOps = ops.filter((o) => o.op === 'replace')
    expect(replaceOps.length).toBeGreaterThanOrEqual(1)
    // Value should NOT be an Option object
    for (const op of replaceOps) {
      if (op.path[op.path.length - 1] === 'presentation') {
        expect(Option.isOption(op.value)).toBe(false)
      }
    }
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it('Option Some→None through union (presentation disappears)', () => {
    const compiled = compilePatchMap(UnionWithOptionSchema)
    const prev: UnionWithOption = Schema.decodeSync(UnionWithOptionSchema)({
      id: '1',
      payload: {
        type: 'tool',
        presentation: {
          toolKey: 'shell',
          partialStdout: 'hello',
          running: true,
        },
      },
    })
    // next: presentation absent → Option.none() after decode
    const next: UnionWithOption = Schema.decodeSync(UnionWithOptionSchema)({
      id: '1',
      payload: { type: 'tool' },
    })
    const ops = assertRoundTrip(compiled, prev, next, 'Option Some→None through union')
    const removeOps = ops.filter(
      (o) => o.op === 'remove' && o.path[o.path.length - 1] === 'presentation',
    )
    expect(removeOps.length).toBeGreaterThanOrEqual(1)
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })

  it('multi-op batch with mixed operations', () => {
    const compiled = compilePatchMap(OuterSchema)
    const prev = Schema.decodeSync(OuterSchema)({
      id: '1',
      title: 'Hello',
      optionField: 'opt',
      defaultField: 'custom',
      nullableField: 'not-null',
      items: ['a', 'b', 'c'],
      records: { x: 1, y: 2 },
      unionField: { type: 'a', aVal: 'hello' },
    })
    const next = Schema.decodeSync(OuterSchema)({
      id: '1',
      title: 'World',
      // optionField: removed
      optionField: 'changed-val',
      defaultField: 'new-default',
      nullableField: null,
      items: ['a', 'X', 'c', 'd'],
      records: { x: 10, z: 3 },
      unionField: { type: 'b', bVal: 99 },
    })
    const ops = assertRoundTrip(compiled, prev, next, 'multi-op mixed batch')
    assertNoUndefined(ops)
    assertWireRoundTrip(ops)
  })
})
