import { describe, it, expect } from 'vitest'
import { Effect, Stream, Duration, Fiber, Scope } from 'effect'
import { createBoundModel } from '../resolver/pipeline'
import type { ExecutableDriver, DriverRequest, StreamResult, CompleteResult } from '../drivers/types'
import { CollectorData } from '../drivers/types'
import type { ProviderModel } from '../model/model'
import { ModelConnection } from '../model/model-connection'
import type { ProviderStateShape } from '../runtime/contracts'
import { ProviderState } from '../runtime/contracts'
import { TraceEmitter } from '../resolver/tracing'
import type { CallUsage } from '../state/provider-state'
import { ModelError, TransportError } from '../errors/model-error'

const nullUsage: CallUsage = {
  inputTokens: null,
  outputTokens: null,
  cacheReadTokens: null,
  cacheWriteTokens: null,
  inputCost: null,
  outputCost: null,
  totalCost: null,
}

/**
 * Creates a fake driver whose stream hangs forever on .next() unless
 * the AbortSignal fires, in which case it rejects immediately.
 */
function createHangingDriver(): ExecutableDriver {
  return {
    id: 'baml',
    connect: (model) => Effect.succeed(ModelConnection.Baml({ auth: null })),
    stream: (req: DriverRequest): Effect.Effect<StreamResult, ModelError> =>
      Effect.succeed({
        stream: Stream.fromAsyncIterable(
          {
            [Symbol.asyncIterator]() {
              return {
                async next() {
                  // If signal exists and is wired, wait for abort then reject
                  if (req.signal) {
                    if (req.signal.aborted) throw new Error('aborted')
                    await new Promise<never>((_, reject) => {
                      req.signal!.addEventListener('abort', () => reject(new Error('aborted')), { once: true })
                    })
                  }
                  // No signal — hang forever (the bug case)
                  return new Promise<never>(() => {})
                },
                async return() {
                  return { done: true as const, value: undefined }
                },
              }
            },
          },
          () => new Error('stream error'),
        ).pipe(
          Stream.map(() => ''),
          Stream.mapError(() => new TransportError({ message: 'stream error', status: null })),
        ),
        getUsage: () => nullUsage,
        getCollectorData: () => CollectorData.Baml({ rawRequestBody: null, rawResponseBody: null, sseEvents: null }),
      }),
    complete: <T = unknown>(req: DriverRequest): Effect.Effect<CompleteResult<T>, ModelError> =>
      Effect.die('not used in this test'),
  }
}

function createProviderState(): ProviderStateShape<string> {
  return {
    peek: () => Effect.succeed(null),
    getSlot: () => Effect.die('not used'),
    setSelection: () => Effect.succeed(false),
    clear: () => Effect.void,
    contextWindow: () => Effect.succeed(0),
    contextLimits: () => Effect.succeed({ hardCap: 0, softCap: 0 }),
    accumulateUsage: () => Effect.void,
    getUsage: () => Effect.die('not used'),
    resetUsage: () => Effect.void,
  }
}

const traceEmitter: { emit: () => Effect.Effect<void>; debug: boolean } = { emit: () => Effect.void, debug: false }

describe('provider stream interrupt cancellation', () => {
  it('interrupting consumer fiber should cancel hanging provider stream promptly', async () => {
    const model: ProviderModel = {
      id: 'test-model',
      providerId: 'test-provider',
      providerName: 'Test Provider',
      name: 'Test Model',
      modelId: null,
      contextWindow: 100_000,
      maxContextTokens: null,
      maxOutputTokens: 8_192,
      supportsToolCalls: false,
      supportsReasoning: false,
      supportsVision: true,
      costs: null,
    }

    const connection = ModelConnection.Baml({ auth: null })
    const driver = createHangingDriver()

    const bound = await Effect.runPromise(
      createBoundModel('primary', model, connection, driver).pipe(
        Effect.provideService(ProviderState, createProviderState()),
        Effect.provideService(TraceEmitter, traceEmitter),
      ),
    )

    const program = Effect.gen(function* () {
      // Fork the entire stream acquisition + consumption so we can interrupt it
      const fiber = yield* Effect.gen(function* () {
        const chatStream = yield* bound.stream('CodingAgentChat', [{}])
        yield* chatStream.stream.pipe(Stream.runDrain)
      }).pipe(Effect.fork)

      // Give the stream time to start and block on .next()
      yield* Effect.sleep(Duration.millis(50))
      // Interrupt — this should abort the underlying transport promptly
      yield* Fiber.interrupt(fiber)
      yield* Fiber.await(fiber)
    }).pipe(
      Effect.provideService(TraceEmitter, traceEmitter),
      // If this times out, the interrupt didn't cancel the stream promptly
      Effect.timeout(Duration.seconds(1)),
    )

    // Success = completes within 1 second without TimeoutException
    await Effect.runPromise(program)
  })

  it('normal completion should preserve late usage finalization (no cleanup abort)', async () => {
    const model: ProviderModel = {
      id: 'test-model',
      providerId: 'test-provider',
      providerName: 'Test Provider',
      name: 'Test Model',
      modelId: null,
      contextWindow: 100_000,
      maxContextTokens: null,
      maxOutputTokens: 8_192,
      supportsToolCalls: false,
      supportsReasoning: false,
      supportsVision: false,
      costs: null,
    }

    const connection = ModelConnection.Baml({ auth: null })

    let finalizedUsage: CallUsage = nullUsage
    const lateUsageDriver: ExecutableDriver = {
      id: 'baml',
      connect: () => Effect.succeed(connection),
      stream: (req: DriverRequest): Effect.Effect<StreamResult, ModelError> =>
        Effect.succeed({
          stream: Stream.unwrapScoped(
            Effect.gen(function* () {
              yield* Effect.addFinalizer(() =>
                Effect.sync(() => {
                  finalizedUsage = req.signal?.aborted
                    ? nullUsage
                    : {
                      inputTokens: 123,
                      outputTokens: 45,
                      cacheReadTokens: null,
                      cacheWriteTokens: null,
                      inputCost: null,
                      outputCost: null,
                      totalCost: null,
                    }
                }),
              )
              return Stream.make('chunk')
            }),
          ),
          getUsage: () => finalizedUsage,
          getCollectorData: () => CollectorData.Baml({ rawRequestBody: null, rawResponseBody: null, sseEvents: null }),
        }),
      complete: <T = unknown>() => Effect.die('not used in this test') as Effect.Effect<CompleteResult<T>, ModelError>,
    }

    const bound = await Effect.runPromise(
      createBoundModel('primary', model, connection, lateUsageDriver).pipe(
        Effect.provideService(ProviderState, createProviderState()),
        Effect.provideService(TraceEmitter, traceEmitter),
      ),
    )

    const chatStream = await Effect.runPromise(
      bound.stream('CodingAgentChat', [{}]).pipe(
        Effect.provideService(TraceEmitter, traceEmitter),
      ),
    )

    await Effect.runPromise(Stream.runDrain(chatStream.stream))

    expect(chatStream.getUsage().inputTokens).toBe(123)
  })
})
