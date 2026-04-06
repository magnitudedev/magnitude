import { Effect, Layer, Stream } from 'effect'

import type { CallUsage } from '../state/provider-state'
import { Model } from '../model/model'
import { ModelConnection } from '../model/model-connection'
import { ModelResolver } from './model-resolver'
import { makeTestTracer, type TraceEmitter, type TracePersister } from './tracing'
import type { BamlFunctionName, BamlResult, BamlStreamFunctionName } from '../drivers/baml-types'
import type { BoundModel, ChatStream, CompleteResult, ModelFunctionDef } from '../model/bound-model'
import type { CollectorData } from '../drivers/types'
import type { ModelError } from '../errors/model-error'

export interface TestModelConfig {
  completeResponse?: string | ((functionName: string, args: readonly unknown[]) => unknown)
  streamResponse?: string | ((functionName: string, args: readonly unknown[]) => string)
}

const fakeModel = new Model({
  id: 'test-model',
  providerId: 'test',
  name: 'Test Model',
  contextWindow: 200_000,
  maxOutputTokens: null,
  costs: null,
})

const zeroUsage: CallUsage = {
  inputTokens: 0,
  outputTokens: 0,
  cacheReadTokens: null,
  cacheWriteTokens: null,
  inputCost: null,
  outputCost: null,
  totalCost: null,
}

const collectorData: CollectorData = {
  _tag: 'Baml',
  rawRequestBody: null,
  rawResponseBody: null,
}

function resolveCompleteResponse(
  config: TestModelConfig,
  functionName: string,
  args: readonly unknown[],
): unknown {
  const value = config.completeResponse
  if (typeof value === 'function') {
    return value(functionName, args)
  }
  return value ?? ''
}

function resolveStreamResponse(
  config: TestModelConfig,
  functionName: string,
  args: readonly unknown[],
): string {
  const value = config.streamResponse
  if (typeof value === 'function') {
    return value(functionName, args)
  }
  return value ?? ''
}

let lastTestTraceStore: ReturnType<typeof makeTestTracer>['store'] | null = null

export function getTestTraceStore() {
  return lastTestTraceStore
}

export function makeTestResolver(config: TestModelConfig = {}): Layer.Layer<ModelResolver | TraceEmitter | TracePersister> {
  const { layer: tracerLayer, store } = makeTestTracer()
  lastTestTraceStore = store

  const fakeBoundModel: BoundModel = {
    model: fakeModel,
    connection: ModelConnection.Baml({
      auth: null,
    }),
    stream: <K extends BamlStreamFunctionName>(functionName: K, args: readonly unknown[]) =>
      Effect.succeed({
        stream: Stream.make(resolveStreamResponse(config, functionName, args)),
        getUsage: () => zeroUsage,
        getCollectorData: () => collectorData,
      } satisfies ChatStream),
    complete: <K extends BamlFunctionName>(functionName: K, args: readonly unknown[]) =>
      Effect.succeed({
        result: resolveCompleteResponse(config, functionName, args) as BamlResult<K>,
        usage: zeroUsage,
      } satisfies CompleteResult<BamlResult<K>>) as Effect.Effect<CompleteResult<BamlResult<K>>, ModelError>,
    invoke<I, O>(fn: ModelFunctionDef<I, O>, input: I) {
      return fn.execute(fakeBoundModel, input)
    },
  }

  return Layer.merge(
    Layer.succeed(ModelResolver, {
      resolve: (_slot: string) => Effect.succeed(fakeBoundModel),
    }),
    tracerLayer,
  )
}