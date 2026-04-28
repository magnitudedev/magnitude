/**
 * TurnEngine unit tests.
 *
 * Uses a mock codec + driver bundled into a NativeTransport via
 * `makeNativeTransport`. Verifies that TurnEngine.runTurn correctly
 * encodes, sends, decodes, and propagates ResponseStreamEvents.
 */

import { describe, it, expect } from 'vitest'
import { Effect, Layer, Stream } from 'effect'
import { FetchHttpClient } from '@effect/platform'
import {
  TurnEngine,
  TurnEngineLive,
  TurnEngineError,
} from '../../src/engine/turn-engine'
import {
  makeNativeTransport,
  type NativeBoundModel,
} from '../../src/engine/native-bound-model'
import {
  CodecEncodeError,
  type Codec,
  type ResponseStreamEvent,
  type ToolDef,
} from '@magnitudedev/codecs'
import {
  DriverError,
  type Driver,
} from '@magnitudedev/drivers'
import type { Message } from '../../src/projections/memory'

// ─── Helpers ────────────────────────────────────────────────────────────────

interface FakeRequest { readonly fake: true }
interface FakeChunk   { readonly text: string }

interface MakeModelOpts {
  readonly events?:       readonly ResponseStreamEvent[]
  readonly encodeError?:  string
  readonly driverError?:  string
}

function makeModel(opts: MakeModelOpts = {}): NativeBoundModel {
  const events      = opts.events ?? []
  const encodeError = opts.encodeError
  const driverError = opts.driverError

  const codec: Codec<FakeRequest, FakeChunk> = {
    id: 'fake-codec',
    encode: () =>
      encodeError !== undefined
        ? Effect.fail(new CodecEncodeError({ reason: encodeError, context: null }))
        : Effect.succeed({ fake: true } as const),
    decode: (chunks) =>
      chunks.pipe(Stream.flatMap(() => Stream.fromIterable(events))),
  }

  // Explicit return-type annotation widens Stream's error channel from
  // `never` (what fromIterable produces) to `DriverError` (what the
  // Driver interface declares). No cast needed — the `never → DriverError`
  // direction is sound since Stream's error channel is covariant for
  // producers.
  const sendOk = (): Effect.Effect<Stream.Stream<FakeChunk, DriverError>, DriverError> =>
    Effect.succeed(Stream.fromIterable<FakeChunk>([{ text: 'chunk' }]))

  const driver: Driver<FakeRequest, FakeChunk> = {
    id: 'fake-driver',
    send: () =>
      driverError !== undefined
        ? Effect.fail(new DriverError({ reason: driverError, status: null, body: null }))
        : sendOk(),
  }

  return {
    model: {
      id: 'test-model',
      providerId: 'test',
      paradigm: 'native',
      maxOutputTokens: 4096,
      supportsReasoning: false,
      supportsVision: false,
    } as never,
    auth:       { type: 'api', key: 'test-key' },
    transport:  makeNativeTransport(codec, driver),
    wireConfig: { endpoint: 'http://localhost', wireModelName: 'model', defaultMaxTokens: 4096 },
  }
}

const memoryEmpty: Message[] = []
const toolsEmpty  = new Map<string, never>()
const toolDefsEmpty: ToolDef[] = []

const baseLayers = Layer.mergeAll(TurnEngineLive, FetchHttpClient.layer)

// ─── Tests ──────────────────────────────────────────────────────────────────

describe('TurnEngine', () => {
  it('happy path — emits ThoughtStart + MessageStart + TurnEnd events', async () => {
    // Codec events use the ResponseStreamEvent shape (no extra `id` field on thought/message)
    const events: ResponseStreamEvent[] = [
      { type: 'thought_start', level: 'medium' },
      { type: 'thought_delta', text: 'planning' },
      { type: 'thought_end' },
      { type: 'message_start' },
      { type: 'message_delta', text: 'hello' },
      { type: 'message_end' },
      { type: 'response_done', reason: 'stop', usage: { inputTokens: 1, outputTokens: 1, cacheReadTokens: null, cacheWriteTokens: null } },
    ]

    const collected = await Effect.gen(function* () {
      const engine = yield* TurnEngine
      const stream = yield* engine.runTurn({
        model:    makeModel({ events }),
        memory:   memoryEmpty,
        tools:    toolsEmpty,
        toolDefs: toolDefsEmpty,
        options:  { thinkingLevel: 'medium' },
      })
      return yield* stream.pipe(Stream.runCollect)
    }).pipe(
      Effect.provide(baseLayers),
      Effect.runPromise,
    )

    const list = Array.from(collected)
    // Response stream events are translated by the engine into TurnEngineEvents.
    expect(list.length).toBeGreaterThanOrEqual(7)
    expect(list[0]._tag).toBe('ThoughtStart')
    expect(list[3]._tag).toBe('MessageStart')
    // Last event (or among last) should be TurnEnd
    const last = list[list.length - 1]
    expect(last._tag).toBe('TurnEnd')
  })

  it('tool call path — emits ToolInputStarted + ToolInputReady + TurnEnd', async () => {
    // Use correct ResponseStreamEvent shapes: toolCallId not id; no tool_call_input_delta
    const events: ResponseStreamEvent[] = [
      { type: 'tool_call_start',       toolCallId: 'tc1', toolName: 'shell' },
      { type: 'tool_call_field_delta', toolCallId: 'tc1', path: ['command'], delta: '"ls"' },
      { type: 'tool_call_end',         toolCallId: 'tc1' },
      { type: 'response_done',         reason: 'tool_calls', usage: { inputTokens: 1, outputTokens: 1, cacheReadTokens: null, cacheWriteTokens: null } },
    ]

    const collected = await Effect.gen(function* () {
      const engine = yield* TurnEngine
      const stream = yield* engine.runTurn({
        model:    makeModel({ events }),
        memory:   memoryEmpty,
        tools:    toolsEmpty,
        toolDefs: toolDefsEmpty,
        options:  { thinkingLevel: 'medium' },
      })
      return yield* stream.pipe(Stream.runCollect)
    }).pipe(
      Effect.provide(baseLayers),
      Effect.runPromise,
    )

    const list = Array.from(collected)
    expect(list[0]._tag).toBe('ToolInputStarted')
    const toolReady = list.find(e => e._tag === 'ToolInputReady')
    expect(toolReady).toBeDefined()
    const turnEnd = list.find(e => e._tag === 'TurnEnd')
    expect(turnEnd).toBeDefined()
  })

  it('driver error → Effect-level TurnEngineError', async () => {
    const result = await Effect.gen(function* () {
      const engine = yield* TurnEngine
      return yield* engine.runTurn({
        model:    makeModel({ driverError: 'transport blew up' }),
        memory:   memoryEmpty,
        tools:    toolsEmpty,
        toolDefs: toolDefsEmpty,
        options:  { thinkingLevel: 'medium' },
      }).pipe(Effect.either)
    }).pipe(
      Effect.provide(baseLayers),
      Effect.runPromise,
    )

    expect(result._tag).toBe('Left')
    if (result._tag === 'Left') {
      expect(result.left).toBeInstanceOf(TurnEngineError)
      expect(result.left.phase).toBe('send')
    }
  })

  it('encode error → Effect-level TurnEngineError', async () => {
    const result = await Effect.gen(function* () {
      const engine = yield* TurnEngine
      return yield* engine.runTurn({
        model:    makeModel({ encodeError: 'cannot encode' }),
        memory:   memoryEmpty,
        tools:    toolsEmpty,
        toolDefs: toolDefsEmpty,
        options:  { thinkingLevel: 'medium' },
      }).pipe(Effect.either)
    }).pipe(
      Effect.provide(baseLayers),
      Effect.runPromise,
    )

    expect(result._tag).toBe('Left')
    if (result._tag === 'Left') {
      expect(result.left).toBeInstanceOf(TurnEngineError)
      expect(result.left.phase).toBe('encode')
    }
  })
})
