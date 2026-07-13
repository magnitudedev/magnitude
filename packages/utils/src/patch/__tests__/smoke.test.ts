import { describe, it, expect } from 'vitest'
import { Schema, Option, Effect } from 'effect'
import { compilePatchMap, diffDecoded, applyDecodedPatch } from '../index'
import type { CompiledMap } from '../index'

// ---------------------------------------------------------------------------
// Test schemas covering key constructs
// ---------------------------------------------------------------------------

const InnerSchema = Schema.Struct({
  name: Schema.String,
  value: Schema.Number,
})

const OuterSchema = Schema.Struct({
  id: Schema.String,
  title: Schema.String,
  optionField: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  defaultField: Schema.optionalWith(Schema.String, { default: () => 'default-val' }),
  nullableField: Schema.Union(Schema.String, Schema.Null),
  items: Schema.Array(Schema.String),
  records: Schema.Record({ key: Schema.String, value: Schema.Number }),
  inner: Schema.optionalWith(InnerSchema, { as: 'Option', exact: true }),
  unionField: Schema.Union(
    Schema.Struct({ type: Schema.Literal('a'), aVal: Schema.String }),
    Schema.Struct({ type: Schema.Literal('b'), bVal: Schema.Number }),
  ),
})

type Outer = Schema.Schema.Type<typeof OuterSchema>

function checkInvariant(
  compiled: CompiledMap,
  prev: Outer,
  next: Outer,
  _label: string,
): void {
  const ops = Effect.runSync(diffDecoded(prev, next, compiled))
  const result = Effect.runSync(applyDecodedPatch(prev, ops, compiled)) as Outer
  expect(result).toEqual(next)
}

describe('decoded patch pipeline — smoke', () => {
  const compiled = compilePatchMap(OuterSchema)

  const basePrev: Outer = Schema.decodeSync(OuterSchema)({
    id: '1',
    title: 'Hello',
    optionField: 'some-val',
    defaultField: 'custom',
    nullableField: 'not-null',
    items: ['a', 'b', 'c'],
    records: { x: 1, y: 2 },
    inner: { name: 'inner', value: 42 },
    unionField: { type: 'a', aVal: 'hello' },
  })

  it('no change → no ops', () => {
    const ops = Effect.runSync(diffDecoded(basePrev, basePrev, compiled))
    expect(ops).toEqual([])
  })

  it('scalar change', () => {
    const next = { ...basePrev, title: 'World' }
    checkInvariant(compiled, basePrev, next, 'scalar change')
  })

  it('Option Some→Some', () => {
    const next = { ...basePrev, optionField: Option.some('changed') }
    checkInvariant(compiled, basePrev, next, 'Option Some→Some')
  })

  it('Option Some→None', () => {
    const next = { ...basePrev, optionField: Option.none() }
    checkInvariant(compiled, basePrev, next, 'Option Some→None')
  })

  it('Option None→Some', () => {
    const prev = { ...basePrev, optionField: Option.none() }
    const next = { ...basePrev, optionField: Option.some('now-some') }
    checkInvariant(compiled, prev, next, 'Option None→Some')
  })

  it('nested Option inner field change', () => {
    const next = {
      ...basePrev,
      inner: Option.some({ name: 'inner-changed', value: 100 }),
    }
    checkInvariant(compiled, basePrev, next, 'nested Option inner change')
  })

  it('nested Option Some→None', () => {
    const next = { ...basePrev, inner: Option.none() }
    checkInvariant(compiled, basePrev, next, 'nested Option Some→None')
  })

  it('nested Option None→Some', () => {
    const prev = { ...basePrev, inner: Option.none() }
    const next = { ...basePrev, inner: Option.some({ name: 'new', value: 1 }) }
    checkInvariant(compiled, prev, next, 'nested Option None→Some')
  })

  it('default field change', () => {
    const next = { ...basePrev, defaultField: 'new-default' }
    checkInvariant(compiled, basePrev, next, 'default field change')
  })

  it('default field present→absent (default applies)', () => {
    // Remove the key so the default kicks in on decode
    const encodedPrev = Schema.encodeSync(OuterSchema)(basePrev)
    const { defaultField: _drop, ...encodedWithoutDefault } = encodedPrev
    const nextDecoded = Schema.decodeSync(OuterSchema)(encodedWithoutDefault)
    expect(nextDecoded.defaultField).toBe('default-val')
    checkInvariant(compiled, basePrev, nextDecoded, 'default field to default')
  })

  it('array element change', () => {
    const next = { ...basePrev, items: ['a', 'X', 'c'] }
    checkInvariant(compiled, basePrev, next, 'array element change')
  })

  it('array insert at tail', () => {
    const next = { ...basePrev, items: ['a', 'b', 'c', 'd'] }
    checkInvariant(compiled, basePrev, next, 'array insert at tail')
  })

  it('array remove from tail', () => {
    const next = { ...basePrev, items: ['a', 'b'] }
    checkInvariant(compiled, basePrev, next, 'array remove from tail')
  })

  it('record add key', () => {
    const next = { ...basePrev, records: { x: 1, y: 2, z: 3 } }
    checkInvariant(compiled, basePrev, next, 'record add key')
  })

  it('record remove key', () => {
    const next = { ...basePrev, records: { x: 1 } }
    checkInvariant(compiled, basePrev, next, 'record remove key')
  })

  it('record change value', () => {
    const next = { ...basePrev, records: { x: 10, y: 2 } }
    checkInvariant(compiled, basePrev, next, 'record change value')
  })

  it('union member swap', () => {
    const next = {
      ...basePrev,
      unionField: { type: 'b' as const, bVal: 99 },
    }
    checkInvariant(compiled, basePrev, next, 'union member swap')
  })

  it('union same member field change', () => {
    const next = {
      ...basePrev,
      unionField: { type: 'a' as const, aVal: 'changed' },
    }
    checkInvariant(compiled, basePrev, next, 'union same member field change')
  })

  it('nullable field to null', () => {
    const next = { ...basePrev, nullableField: null }
    checkInvariant(compiled, basePrev, next, 'nullable to null')
  })

  it('nullable field from null to value', () => {
    const prev = { ...basePrev, nullableField: null }
    const next = { ...basePrev, nullableField: 'val' }
    checkInvariant(compiled, prev, next, 'nullable from null')
  })

  it('multiple changes at once', () => {
    const next = {
      ...basePrev,
      title: 'multi',
      optionField: Option.some('multi-opt'),
      items: ['x', 'y'],
      records: { w: 0 },
      inner: Option.some({ name: 'multi-inner', value: 200 }),
    }
    checkInvariant(compiled, basePrev, next, 'multiple changes')
  })

  it('structural sharing — unchanged refs preserved', () => {
    const next = { ...basePrev, title: 'new-title' }
    const ops = Effect.runSync(diffDecoded(basePrev, next, compiled))
    const result = Effect.runSync(applyDecodedPatch(basePrev, ops, compiled)) as Outer
    // Unchanged array should keep same reference
    expect(result.items).toBe(basePrev.items)
    expect(result.records).toBe(basePrev.records)
    expect(result.inner).toBe(basePrev.inner)
  })
})

// ---------------------------------------------------------------------------
// Production schema test — DisplayViewSnapshot
// ---------------------------------------------------------------------------

import { DisplayViewSnapshot } from '../../../../protocol/src/schemas/display'
import type { DisplayViewSnapshot as DVS } from '../../../../protocol/src/schemas/display'

describe('decoded patch pipeline — production DisplayViewSnapshot', () => {
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

  it('production path: ToolMessage → presentation Option → ShellPresentation → partialStdout', () => {
    const prev = makeSnapshot('hello')
    const next = makeSnapshot('hello world')
    const ops = Effect.runSync(diffDecoded(prev, next, compiled))
    // Should be a single leaf-level replace — same data as JSON Patch
    expect(ops).toEqual([
      { op: 'replace', path: ['state', 'timelines', 'root', 'messages', 'byId', 'm1', 'presentation', 'partialStdout'], value: 'hello world' },
    ])
    const result = Effect.runSync(applyDecodedPatch(prev, ops, compiled)) as DVS
    expect(result).toEqual(next)
  })

  it('production path: add a new message', () => {
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

    const ops = Effect.runSync(diffDecoded(prev, next, compiled))
    const result = Effect.runSync(applyDecodedPatch(prev, ops, compiled)) as DVS
    expect(result).toEqual(next)
  })

  it('production path: presentation Some→None', () => {
    const prev = makeSnapshot('hello')
    const m1 = prev.state.timelines.root.messages.byId.m1
    const next = {
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

    const ops = Effect.runSync(diffDecoded(prev, next, compiled))
    const result = Effect.runSync(applyDecodedPatch(prev, ops, compiled)) as DVS
    expect(result).toEqual(next)
  })

  it('production path: structural sharing — unchanged messages preserved', () => {
    const prev = makeSnapshot('hello')
    const next = makeSnapshot('hello world')
    const ops = Effect.runSync(diffDecoded(prev, next, compiled))
    const result = Effect.runSync(applyDecodedPatch(prev, ops, compiled)) as DVS
    // Unchanged parts should keep same reference
    expect(result.state.session).toBe(prev.state.session)
    expect(result.state.tasks).toBe(prev.state.tasks)
    expect(result.state.timelines.root.messages.byId.m1.id).toBe('m1')
  })
})
