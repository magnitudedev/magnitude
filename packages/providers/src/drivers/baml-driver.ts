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

import { bamlCall, bamlStream } from './baml-dispatch'
import { logger } from '@magnitudedev/logger'
import { normalizeAnthropicUsage } from './usage-normalization'

function validateTokenCount(tokens: number): number | null {
  if (tokens <= 0) return null
  return tokens
}

function applyAnthropicUsage(rawUsage: unknown): {
  inputTokens: number | null
  outputTokens: number | null
  cacheReadTokens: number | null
  cacheWriteTokens: number | null
} {
  const normalized = normalizeAnthropicUsage(rawUsage)
  return {
    inputTokens: normalized.inputTokens === null ? null : validateTokenCount(normalized.inputTokens),
    outputTokens: normalized.outputTokens,
    cacheReadTokens: normalized.cacheReadTokens,
    cacheWriteTokens: normalized.cacheWriteTokens,
  }
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

function extractUsageFromCollector(
  collector: Collector,
  model: Model | null,
  authType: string | null,
): {
  usage: CallUsage
  diagnostics: {
    usageSource: 'http-response-usage' | 'anthropic-sse-usage' | 'collector-usage' | 'none'
    rawUsage: unknown | null
    parsedInputTokens: number | null
    parsedOutputTokens: number | null
    parsedCacheReadTokens: number | null
    parsedCacheWriteTokens: number | null
    providerId: string | null
    modelId: string | null
    authType: string | null
    driverId: 'baml'
    usageAbsent: boolean
  }
} {
  let inputTokens: number | null = null
  let outputTokens: number | null = null
  let cacheReadTokens: number | null = null
  let cacheWriteTokens: number | null = null
  let usageSource: 'http-response-usage' | 'anthropic-sse-usage' | 'collector-usage' | 'none' = 'none'
  let rawUsage: unknown | null = null

  const lastCall = collector.last?.calls.at(-1)

  if (lastCall) {
    // Strategy 1: Extract from HTTP response body JSON
    try {
      const rawUsageFromHttp = lastCall.httpResponse?.body.json()?.usage
      if (rawUsageFromHttp) {
        rawUsage = rawUsageFromHttp
        usageSource = 'http-response-usage'
        const parsed = applyAnthropicUsage(rawUsageFromHttp)
        if (parsed.inputTokens !== null) inputTokens = parsed.inputTokens
        if (parsed.outputTokens !== null) outputTokens = parsed.outputTokens
        cacheReadTokens = parsed.cacheReadTokens
        cacheWriteTokens = parsed.cacheWriteTokens
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
              rawUsage = usage
              usageSource = 'anthropic-sse-usage'
              const parsed = applyAnthropicUsage(usage)
              if (parsed.inputTokens !== null) inputTokens = parsed.inputTokens
              cacheReadTokens = parsed.cacheReadTokens
              cacheWriteTokens = parsed.cacheWriteTokens
            }
            if (data.type === 'message_delta' && data.usage) {
              const parsed = applyAnthropicUsage(data.usage)
              if (parsed.outputTokens !== null) {
                outputTokens = parsed.outputTokens
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
      rawUsage = usage
      usageSource = 'collector-usage'
      const total = (usage.inputTokens ?? 0) + (usage.cachedInputTokens ?? 0)
      inputTokens = validateTokenCount(total)
    }
  }
  if (outputTokens === null) {
    const usage = collector.usage
    if (usage && typeof usage.outputTokens === 'number') outputTokens = usage.outputTokens
  }

  const usage = buildUsage(model, authType, inputTokens, outputTokens, cacheReadTokens, cacheWriteTokens)
  return {
    usage,
    diagnostics: {
      usageSource,
      rawUsage,
      parsedInputTokens: inputTokens,
      parsedOutputTokens: outputTokens,
      parsedCacheReadTokens: cacheReadTokens,
      parsedCacheWriteTokens: cacheWriteTokens,
      providerId: model?.providerId ?? null,
      modelId: model?.id ?? null,
      authType,
      driverId: 'baml',
      usageAbsent: inputTokens === null && outputTokens === null,
    },
  }
}

function extractCollectorData(
  collector: Collector,
  diagnostics?: {
    usageSource: 'http-response-usage' | 'anthropic-sse-usage' | 'collector-usage' | 'none'
    rawUsage: unknown | null
    parsedInputTokens: number | null
    parsedOutputTokens: number | null
    parsedCacheReadTokens: number | null
    parsedCacheWriteTokens: number | null
    providerId: string | null
    modelId: string | null
    authType: string | null
    driverId: 'baml'
    usageAbsent: boolean
  } | null,
): ReturnType<typeof CollectorData.Baml> {
  const lastCall = collector.last?.calls.at(-1)
  let rawRequestBody: unknown = null
  let rawResponseBody: unknown = null

  try {
    rawRequestBody = lastCall?.httpRequest?.body?.json?.() ?? null
  } catch {}

  try {
    rawResponseBody = lastCall?.httpResponse?.body?.json?.() ?? null
  } catch {}

  if (diagnostics) {
    logger.info(
      { context: 'BamlDriverUsageDiagnostics', ...diagnostics },
      '[BamlDriver] usage diagnostics',
    )
  }

  return CollectorData.Baml({ rawRequestBody, rawResponseBody, diagnostics: diagnostics ?? null })
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
            return extractUsageFromCollector(collector, req.model, authType).usage
          },
          getCollectorData() {
            const extracted = extractUsageFromCollector(collector, req.model, authType)
            return extractCollectorData(collector, extracted.diagnostics)
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
        const opts = { clientRegistry, collector }
        const result = await bamlCall(req.functionName, req.args, opts)
        const authType = req.connection._tag === 'Baml' ? (req.connection.auth?.type ?? null) : null
        const extracted = extractUsageFromCollector(collector, req.model, authType)
        return {
          result: normalizeModelOutput(result) as T,
          usage: extracted.usage,
          collectorData: extractCollectorData(collector, extracted.diagnostics),
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