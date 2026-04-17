import { BamlClientHttpError, isBamlError, Collector, type ClientRegistry } from '@magnitudedev/llm-core'
import { Effect, Stream } from 'effect'
import { ModelConnection as ModelConnectionCtor } from '../model/model-connection'
import type { InferenceConfig } from '../model/inference-config'
import type { ModelError } from '../errors/model-error'

import { normalizeModelOutput, normalizeQuotesInString } from '../util/output-normalization'
import { toIncrementalStream } from '../util/incremental-stream'
import { fromAsyncIterableSafe } from '../util/from-async-iterable-safe'
import { buildUsage } from '../usage'
import { buildClientRegistry } from '../client-registry-builder'
import type { CallUsage } from '../state/provider-state'
import type { ExecutableDriver, DriverRequest, StreamResult, CompleteResult } from './types'
import type { AuthInfo } from '../types'
import type { Model } from '../model/model'
import { CollectorData } from './types'
import { classifyHttpError, classifyUnknownError } from '../errors/classify-error'
import { TransportError } from '../errors/model-error'

import { bamlStream } from './baml-dispatch'

import { extractUsageFromCollectorData } from './usage-extraction'

/** Build a ClientRegistry on demand from the driver request */
function buildRegistry(req: DriverRequest): ClientRegistry | undefined {
  return buildClientRegistry(
    req.model.providerId,
    req.model.id,
    req.connection.auth,
    req.providerOptions,
    req.inference.stopSequences ? [...req.inference.stopSequences] : undefined,
    req.grammar,
    req.inference.maxTokens,
  )
}

function extractUsageFromCollector(
  collector: Collector,
  model: Model | null,
  authType: string | null,
): CallUsage {
  const extracted = extractUsageFromCollectorData(collector as Parameters<typeof extractUsageFromCollectorData>[0])
  return buildUsage(
    model,
    authType,
    extracted.inputTokens,
    extracted.outputTokens,
    extracted.cacheReadTokens,
    extracted.cacheWriteTokens,
  )
}

function extractCollectorData(collector: Collector): ReturnType<typeof CollectorData.Baml> {
  const lastCall = collector.last?.calls.at(-1)
  let rawRequestBody: unknown = null
  let rawResponseBody: unknown = null

  try {
    rawRequestBody = lastCall?.httpRequest?.body?.json?.() ?? null
  } catch {}

  try {
    rawResponseBody = lastCall?.httpResponse?.body?.json?.() ?? null
  } catch {}

  return CollectorData.Baml({ rawRequestBody, rawResponseBody })
}

function toNormalizedAsyncStream(stream: AsyncIterable<string>): AsyncIterable<string> {
  return {
    async *[Symbol.asyncIterator]() {
      for await (const chunk of stream) {
        yield normalizeQuotesInString(chunk)
      }
    },
  }
}

export const BamlDriver: ExecutableDriver = {
  id: 'baml',
  connect(_model: Model, auth: AuthInfo | null, _inference: InferenceConfig): Effect.Effect<ReturnType<typeof ModelConnectionCtor.Baml>, ModelError> {
    return Effect.succeed(ModelConnectionCtor.Baml({ auth }))
  },
  stream(req: DriverRequest) {
    return Effect.tryPromise({
      try: async (): Promise<StreamResult> => {
        if (req.connection._tag !== 'Baml') {
          throw new Error('Invalid connection type for BamlDriver')
        }
        const clientRegistry = buildRegistry(req)
        const collector = new Collector('model-stream')
        const opts = { clientRegistry, collector, signal: req.signal }
        const bamlStreamResult = bamlStream(req.functionName, req.args, opts)
        const authType = req.connection._tag === 'Baml' ? (req.connection.auth?.type ?? null) : null
        const asyncIter = toNormalizedAsyncStream(toIncrementalStream(bamlStreamResult))
        return {
          stream: fromAsyncIterableSafe(asyncIter, (e) => {
            if (e instanceof BamlClientHttpError) {
              const text = [e.message, e.detailed_message, e.raw_response]
                .filter(Boolean)
                .join(' ')
              return classifyHttpError(e.status_code, text)
            }
            // BAML config/construction/validation errors are non-retryable
            if (isBamlError(e)) {
              return new TransportError({ message: e instanceof Error ? e.message : String(e), status: 400 })
            }
            return classifyUnknownError(e)
          }),
          getUsage(): CallUsage {
            return extractUsageFromCollector(collector, req.model, authType)
          },
          getCollectorData() {
            return extractCollectorData(collector)
          },
        }
      },
      catch: (error) => {
        if (error instanceof BamlClientHttpError) {
          const text = [error.message, error.detailed_message, error.raw_response]
            .filter(Boolean)
            .join(' ')
          return classifyHttpError(error.status_code, text)
        }
        // BAML config/construction/validation errors are non-retryable
        if (isBamlError(error)) {
          return new TransportError({ message: error instanceof Error ? error.message : String(error), status: 400 })
        }
        return classifyUnknownError(error)
      },
    })
  },
  complete<T = unknown>(req: DriverRequest) {
    return Effect.tryPromise({
      try: async (): Promise<CompleteResult<T>> => {
        if (req.connection._tag !== 'Baml') {
          throw new Error('Invalid connection type for BamlDriver')
        }
        const clientRegistry = buildRegistry(req)
        const collector = new Collector(`${req.functionName}-complete`)
        const authType = req.connection._tag === 'Baml' ? (req.connection.auth?.type ?? null) : null
        const opts = { clientRegistry, collector, signal: req.signal }
        const stream = bamlStream(req.functionName, req.args, opts)
        const result = await stream.getFinalResponse()
        return {
          result: normalizeModelOutput(result) as T,
          usage: extractUsageFromCollector(collector, req.model, authType),
          collectorData: extractCollectorData(collector),
        }
      },
      catch: (error) => {
        if (error instanceof BamlClientHttpError) {
          const text = [error.message, error.detailed_message, error.raw_response]
            .filter(Boolean)
            .join(' ')
          return classifyHttpError(error.status_code, text)
        }
        // BAML config/construction/validation errors are non-retryable
        if (isBamlError(error)) {
          return new TransportError({ message: error instanceof Error ? error.message : String(error), status: 400 })
        }
        return classifyUnknownError(error)
      },
    })
  },
}