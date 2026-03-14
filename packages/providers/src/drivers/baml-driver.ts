import { BamlClientHttpError, Collector, type ClientRegistry } from '@magnitudedev/llm-core'
import { Effect, Stream } from 'effect'
import { ModelConnection as ModelConnectionCtor } from '../model/model-connection'
import type { InferenceConfig } from '../model/inference-config'
import type { ModelError } from '../errors/model-error'

import { normalizeModelOutput, normalizeQuotesInString } from '../util/output-normalization'
import { toIncrementalStream } from '../util/incremental-stream'
import { buildUsage } from '../usage'
import { buildClientRegistry } from '../client-registry-builder'
import type { CallUsage } from '../state/provider-state'
import type { ExecutableDriver, DriverRequest, StreamResult, CompleteResult } from './types'
import type { AuthInfo } from '../types'
import type { Model } from '../model/model'
import { CollectorData } from './types'
import { classifyHttpError, classifyUnknownError } from '../errors/classify-error'

import { bamlCall, bamlStream } from './baml-dispatch'

function validateTokenCount(tokens: number): number | null {
  if (tokens <= 0) return null
  return tokens
}

/** Build a ClientRegistry on demand from the driver request */
function buildRegistry(req: DriverRequest): ClientRegistry | undefined {
  return buildClientRegistry(
    req.model.providerId,
    req.model.id,
    req.connection.auth,
    req.providerOptions,
    req.inference.stopSequences ? [...req.inference.stopSequences] : undefined,
  )
}

function extractUsageFromCollector(collector: Collector, model: Model | null, authType: string | null): CallUsage {
  let inputTokens: number | null = null
  let outputTokens: number | null = null
  let cacheReadTokens: number | null = null
  let cacheWriteTokens: number | null = null

  const lastCall = collector.last?.calls.at(-1)

  if (lastCall) {
    // Strategy 1: Extract from HTTP response body JSON
    try {
      const rawUsage = lastCall.httpResponse?.body.json()?.usage
      if (rawUsage) {
        if (typeof rawUsage.input_tokens === 'number') {
          const total = rawUsage.input_tokens
            + (rawUsage.cache_creation_input_tokens ?? 0)
            + (rawUsage.cache_read_input_tokens ?? 0)
          inputTokens = validateTokenCount(total)
          cacheReadTokens = rawUsage.cache_read_input_tokens ?? null
          cacheWriteTokens = rawUsage.cache_creation_input_tokens ?? null
        }
        if (typeof rawUsage.output_tokens === 'number') {
          outputTokens = rawUsage.output_tokens
        }
      }
    } catch {}

    // Strategy 2: Extract from SSE events (Anthropic streaming)
    if (inputTokens === null) {
      try {
        // sseResponses() is present on streaming collector calls but not in the base type
        const sseResponses = 'sseResponses' in lastCall
          ? (lastCall as { sseResponses(): Array<{ json?(): unknown }> }).sseResponses()
          : null
        if (Array.isArray(sseResponses)) {
          for (const sse of sseResponses) {
            // SSE data is parsed JSON from the provider — untyped by nature
            const data = sse.json?.() as { type?: string; message?: { usage?: any }; usage?: any } | null
            if (!data) continue

            if (data.type === 'message_start' && data.message?.usage) {
              const usage = data.message.usage
              if (typeof usage.input_tokens === 'number') {
                const total = usage.input_tokens
                  + (usage.cache_creation_input_tokens ?? 0)
                  + (usage.cache_read_input_tokens ?? 0)
                inputTokens = validateTokenCount(total)
                cacheReadTokens = usage.cache_read_input_tokens ?? null
                cacheWriteTokens = usage.cache_creation_input_tokens ?? null
              }
            }
            if (data.type === 'message_delta' && data.usage) {
              if (typeof data.usage.output_tokens === 'number') {
                outputTokens = data.usage.output_tokens
              }
            }
          }
        }
      } catch {}
    }
  }

  // Strategy 3: Fallback to collector-level usage
  if (inputTokens === null) {
    const usage = collector.usage
    if (usage) {
      const total = (usage.inputTokens ?? 0) + (usage.cachedInputTokens ?? 0)
      inputTokens = validateTokenCount(total)
    }
  }
  if (outputTokens === null) {
    const usage = collector.usage
    if (usage && typeof usage.outputTokens === 'number') outputTokens = usage.outputTokens
  }

  return buildUsage(model, authType, inputTokens, outputTokens, cacheReadTokens, cacheWriteTokens)
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
  connect(model: Model, auth: AuthInfo | null, _inference: InferenceConfig): Effect.Effect<ReturnType<typeof ModelConnectionCtor.Baml>, ModelError> {
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
        const opts = { clientRegistry, collector }
        const bamlStreamResult = bamlStream(req.functionName, req.args, opts)
        const authType = req.connection._tag === 'Baml' ? (req.connection.auth?.type ?? null) : null
        const asyncIter = toNormalizedAsyncStream(toIncrementalStream(bamlStreamResult))
        return {
          stream: Stream.fromAsyncIterable(asyncIter, (e) => classifyUnknownError(e)),
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
        const opts = { clientRegistry, collector }
        const result = await bamlCall(req.functionName, req.args, opts)
        const authType = req.connection._tag === 'Baml' ? (req.connection.auth?.type ?? null) : null
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
        return classifyUnknownError(error)
      },
    })
  },
}