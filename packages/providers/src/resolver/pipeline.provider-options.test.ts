import { describe, expect, it } from 'vitest'
import { Effect, Stream } from 'effect'
import { Model } from '../model/model'
import { ModelConnection } from '../model/model-connection'
import { createBoundModel } from './pipeline'
import { ProviderState } from '../runtime/contracts'
import { TraceEmitter } from './tracing'
import type { ExecutableDriver } from '../drivers/types'

describe('createBoundModel provider option overrides', () => {
  const traceEmitterStub = {
    emit: () => Effect.void,
  }
  const model = new Model({
    id: 'gpt-5.4',
    providerId: 'openai',
    name: 'gpt-5.4',
    contextWindow: 200000,
    maxOutputTokens: null,
    costs: null,
  })

  const connection = ModelConnection.Baml({ auth: null })

  const providerStateStub = {
    peek: () => Effect.succeed(null),
    getSlot: () => Effect.die('unused'),
    setSelection: () => Effect.succeed(false),
    clear: () => Effect.void,
    contextWindow: () => Effect.succeed(0),
    contextLimits: () => Effect.succeed({ hardCap: 0, softCap: 0 }),
    accumulateUsage: () => Effect.void,
    getUsage: () => Effect.die('unused'),
    resetUsage: () => Effect.void,
  } as any

  it('merges static and per-call providerOptions in stream and complete', async () => {
    let streamProviderOptions: Record<string, unknown> | undefined
    let completeProviderOptions: Record<string, unknown> | undefined

    const driver: ExecutableDriver = {
      id: 'baml',
      connect: () => Effect.succeed(connection),
      stream: (req) => Effect.succeed({
        stream: Stream.make('ok'),
        getUsage: () => ({
          inputTokens: null,
          outputTokens: null,
          cacheReadTokens: null,
          cacheWriteTokens: null,
          inputCost: null,
          outputCost: null,
          totalCost: null,
        }),
        getCollectorData: () => {
          streamProviderOptions = req.providerOptions as Record<string, unknown> | undefined
          return { _tag: 'Baml', rawRequestBody: null, rawResponseBody: null }
        },
      }),
      complete: (req) => {
        completeProviderOptions = req.providerOptions as Record<string, unknown> | undefined
        return Effect.succeed({
          result: 'ok',
          usage: {
            inputTokens: null,
            outputTokens: null,
            cacheReadTokens: null,
            cacheWriteTokens: null,
            inputCost: null,
            outputCost: null,
            totalCost: null,
          },
          collectorData: { _tag: 'Baml', rawRequestBody: null, rawResponseBody: null },
        })
      },
    }

    const bound = await Effect.runPromise(
      createBoundModel('main', model, connection, driver, {}, { baseUrl: 'https://base', instructions: 'static' }).pipe(
        Effect.provideService(ProviderState, providerStateStub),
      ),
    )

    const streamResult = await Effect.runPromise(
      bound.stream('CodingAgentChat' as any, [], { providerOptions: { instructions: 'dynamic' } }).pipe(
        Effect.provideService(TraceEmitter, traceEmitterStub),
      ),
    )
    await Effect.runPromise(Stream.runDrain(streamResult.stream))
    streamResult.getCollectorData()

    await Effect.runPromise(
      bound.complete('GenerateChatTitle' as any, [], { providerOptions: { instructions: 'dynamic-2' } }).pipe(
        Effect.provideService(TraceEmitter, traceEmitterStub),
      ),
    )

    expect(streamProviderOptions).toEqual({ baseUrl: 'https://base', instructions: 'dynamic' })
    expect(completeProviderOptions).toEqual({ baseUrl: 'https://base', instructions: 'dynamic-2' })
  })
})
