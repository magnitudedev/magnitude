import { describe, expect, it } from 'vitest'
import { Deferred, Effect, Fiber, Stream } from 'effect'
import * as HttpClient from '@effect/platform/HttpClient'
import {
  PromptBuilder,
  StreamStartOperationalFailure,
  StreamStartProviderRejection,
  type BaseCallOptions,
  type BoundModel,
  type ModelStreamResult,
} from '@magnitudedev/ai'
import { makeAgentBoundModel } from '../src/model/agent-model'
import { ModelRequestPreparationFailed } from '../src/model/model-request-preparation'

const prompt = PromptBuilder.empty().user('hello').build()
const result: ModelStreamResult = {
  events: Stream.empty,
  parsers: new Map(),
  logprobs: [],
  requestId: null,
}

const config = {
  modelSource: { slotId: 'primary' as const },
  modelId: 'model',
  modelDisplayName: 'Model',
  providerId: 'local',
  profile: { contextWindow: 8192, maxOutputTokens: 1024 },
  debug: false,
  agentId: 'agent',
}

describe('agent model request preparation', () => {
  it('holds preparation through provider acceptance and then releases it', async () => {
    const order: string[] = []
    const rawModel: BoundModel<BaseCallOptions> = {
      stream: () => Effect.sync(() => {
        order.push('provider-accepted')
        return result
      }),
    }
    const model = makeAgentBoundModel({
      ...config,
      rawModel,
      prepareRequest: Effect.acquireRelease(
        Effect.sync(() => {
          order.push('prepared')
        }),
        () => Effect.sync(() => {
          order.push('released')
        }),
      ),
      clearRequestProgress: Effect.sync(() => {
        order.push('cleared')
      }),
    })

    await Effect.runPromise(model.model.stream(prompt, []).pipe(
      Effect.provideService(HttpClient.HttpClient, {} as HttpClient.HttpClient),
    ))

    expect(order).toEqual(['prepared', 'provider-accepted', 'released'])
  })

  it('does not invoke the provider when preparation fails', async () => {
    let providerCalled = false
    let cleared = false
    const rawModel: BoundModel<BaseCallOptions> = {
      stream: () => Effect.sync(() => {
        providerCalled = true
        return result
      }),
    }
    const expected = new ModelRequestPreparationFailed({
      code: 'low_memory',
      message: 'Not enough memory',
      retryable: true,
    })
    const model = makeAgentBoundModel({
      ...config,
      rawModel,
      prepareRequest: Effect.fail(expected),
      clearRequestProgress: Effect.sync(() => {
        cleared = true
      }),
    })

    const failure = await Effect.runPromise(Effect.flip(model.model.stream(prompt, []).pipe(
      Effect.provideService(HttpClient.HttpClient, {} as HttpClient.HttpClient),
    )))

    expect(failure).toBe(expected)
    expect(providerCalled).toBe(false)
    expect(cleared).toBe(true)
  })

  it('releases preparation and clears progress when provider start fails', async () => {
    const order: string[] = []
    const failure = new StreamStartOperationalFailure({
      call: {
        provider: 'local',
        model: 'model',
        method: 'POST',
        url: 'icn://chat/model',
      },
      reason: {
        _tag: 'RequestFailedBeforeResponse',
        cause: {
          _tag: 'ErrorCause',
          name: 'Error',
          message: 'connection lost',
        },
      },
    })
    const rawModel: BoundModel<BaseCallOptions> = {
      stream: () => Effect.fail(failure),
    }
    const model = makeAgentBoundModel({
      ...config,
      rawModel,
      prepareRequest: Effect.acquireRelease(
        Effect.sync(() => {
          order.push('prepared')
        }),
        () => Effect.sync(() => {
          order.push('released')
        }),
      ),
      clearRequestProgress: Effect.sync(() => {
        order.push('cleared')
      }),
    })

    const observed = await Effect.runPromise(Effect.flip(model.model.stream(prompt, []).pipe(
      Effect.provideService(HttpClient.HttpClient, {} as HttpClient.HttpClient),
    )))

    expect(observed).toBe(failure)
    expect(order).toEqual(['prepared', 'released', 'cleared'])
  })

  it('releases preparation and clears progress when start is interrupted', async () => {
    const order = await Effect.runPromise(Effect.gen(function* () {
      const acquired = yield* Deferred.make<void>()
      const order: string[] = []
      const rawModel: BoundModel<BaseCallOptions> = {
        stream: () => Effect.never,
      }
      const model = makeAgentBoundModel({
        ...config,
        rawModel,
        prepareRequest: Effect.acquireRelease(
          Deferred.succeed(acquired, undefined),
          () => Effect.sync(() => {
            order.push('released')
          }),
        ),
        clearRequestProgress: Effect.sync(() => {
          order.push('cleared')
        }),
      })

      const fiber = yield* Effect.fork(model.model.stream(prompt, []).pipe(
        Effect.provideService(HttpClient.HttpClient, {} as HttpClient.HttpClient),
      ))
      yield* Deferred.await(acquired)
      yield* Fiber.interrupt(fiber)
      return order
    }))

    expect(order).toEqual(['released', 'cleared'])
  })

  it('re-prepares exactly once when local admission races an idle unload', async () => {
    let preparations = 0
    let starts = 0
    const notReady = new StreamStartProviderRejection({
      call: {
        provider: 'local',
        model: 'model',
        method: 'POST',
        url: 'icn://chat/model',
      },
      response: {
        status: 409,
        headers: [],
        body: JSON.stringify({
          error: {
            type: 'invalid_request_error',
            code: 'model_not_ready',
            message: 'model is releasing',
          },
        }),
        requestId: null,
        retryAfterMs: null,
      },
      rejection: {
        _tag: 'ModelUnavailable',
        message: 'model is releasing',
      },
    })
    const rawModel: BoundModel<BaseCallOptions> = {
      stream: () => Effect.suspend(() => {
        starts += 1
        return starts === 1 ? Effect.fail(notReady) : Effect.succeed(result)
      }),
    }
    const model = makeAgentBoundModel({
      ...config,
      rawModel,
      prepareRequest: Effect.sync(() => {
        preparations += 1
      }),
    })

    await Effect.runPromise(model.model.stream(prompt, []).pipe(
      Effect.provideService(HttpClient.HttpClient, {} as HttpClient.HttpClient),
    ))

    expect(preparations).toBe(2)
    expect(starts).toBe(2)
  })
})
