import { describe, expect, it } from 'vitest'
import { Effect, Stream } from 'effect'
import { createBoundModel } from './pipeline'
import type { ProviderModel, ModelCosts } from '../model/model'
import { ModelConnection } from '../model/model-connection'
import type { ExecutableDriver } from '../drivers/types'
import { CollectorData } from '../drivers/types'
import { ProviderState } from '../runtime/contracts'
import { TraceEmitter } from './tracing'

const model: ProviderModel = {
  id: 'gpt-5.3-codex',
  providerId: 'openai',
  providerName: 'OpenAI',
  name: 'test',
  modelId: null,
  contextWindow: 100_000,
  maxContextTokens: null,
  maxOutputTokens: 8192,
  supportsToolCalls: false,
  supportsReasoning: false,
  supportsVision: true,
  costs: null as unknown as ModelCosts,
}

const providerState = {
  peek: () => Effect.succeed(null),
  getSlot: () => Effect.die('unused'),
  setSelection: () => Effect.succeed(false),
  clear: () => Effect.void,
  contextWindow: () => Effect.succeed(0),
  contextLimits: () => Effect.succeed({ hardCap: 0, softCap: 0 }),
  accumulateUsage: () => Effect.void,
  getUsage: () => Effect.die('unused'),
  resetUsage: () => Effect.void,
}

const traceEmitter = { emit: () => Effect.void, debug: false }

describe('pipeline providerOptions overrides', () => {
  it('merges per-call providerOptions over bound providerOptions for stream and complete', async () => {
    const seen: Array<Record<string, unknown> | undefined> = []

    const driver: ExecutableDriver = {
      id: 'baml',
      connect: () => Effect.succeed(ModelConnection.Baml({ auth: null })),
      stream: (req) => {
        seen.push(req.providerOptions as Record<string, unknown> | undefined)
        return Effect.succeed({
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
          getCollectorData: () => CollectorData.Baml({ rawRequestBody: null, rawResponseBody: null, sseEvents: null }),
        })
      },
      complete: (req) => {
        seen.push(req.providerOptions as Record<string, unknown> | undefined)
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
          collectorData: CollectorData.Baml({ rawRequestBody: null, rawResponseBody: null, sseEvents: null }),
        })
      },
    }

    const bound = await Effect.runPromise(
      createBoundModel(
        'primary',
        model,
        ModelConnection.Baml({ auth: null }),
        driver,
        {},
        { instructions: 'base', rememberedModelIds: ['keep'] },
      ).pipe(
        Effect.provideService(ProviderState, providerState as any),
      ),
    )

    await Effect.runPromise(
      bound.stream('SimpleChat', ['system', []], { providerOptions: { instructions: 'call', store: false } }).pipe(
        Effect.provideService(TraceEmitter, traceEmitter as any),
      ),
    )

    await Effect.runPromise(
      bound.complete('ExtractMemoryDiff', ['transcript', 'memory'], {
        providerOptions: { instructions: 'call2', headers: { 'X-Test': '1' } },
      }).pipe(Effect.provideService(TraceEmitter, traceEmitter as any)),
    )

    expect(seen[0]).toMatchObject({ instructions: 'call', store: false, rememberedModelIds: ['keep'] })
    expect(seen[1]).toMatchObject({ instructions: 'call2', headers: { 'X-Test': '1' }, rememberedModelIds: ['keep'] })
  })
})
