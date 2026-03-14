import { Effect, Stream, Schedule, Duration, Scope, Exit, Option, Sink } from 'effect'
import type { Model } from '../model/model'
import type { ModelConnection } from '../model/model-connection'
import type { InferenceConfig } from '../model/inference-config'
import type { ExecutableDriver } from '../drivers/types'
import type { BamlFunctionName, BamlResult, BamlStreamFunctionName } from '../drivers/baml-types'
import type { BoundModel, ChatStream, CompleteOptions, CompleteResult, ModelFunctionDef, StreamOptions } from '../model/bound-model'
import type { CallUsage } from '../state/provider-state'
import { CollectorData } from '../drivers/types'
import type { ProviderOptions } from '../types'
import { logger } from '@magnitudedev/logger'
import type { ModelError } from '../errors/model-error'
import { isRetryableError } from '../errors/classify-error'

import type { ModelSlot } from '../state/provider-state'
import { ProviderState, type ProviderStateShape } from '../runtime/contracts'
import { TraceEmitter } from './tracing'

/** Retry schedule for transient connection failures before first chunk */
const connectionRetrySchedule = Schedule.exponential(Duration.seconds(1), 1.5).pipe(
  Schedule.jittered,
  Schedule.intersect(Schedule.recurs(6)),
)



function extractTraceRequest(
  collectorData: CollectorData,
  fallback: { input: unknown },
): { messages?: unknown[]; input?: unknown } {
  const raw = collectorData.rawRequestBody as Record<string, unknown> | null | undefined
  if (!raw) return fallback
  // BAML driver: raw request has `messages` array
  if (Array.isArray(raw.messages)) return { messages: raw.messages }
  // Responses driver: BAML request body uses `input` instead of `messages`
  if (Array.isArray(raw.input)) return { messages: raw.input }
  return fallback
}

function extractTraceResponse(
  collectorData: CollectorData,
  rawOutput: string | null,
): { rawBody: unknown | null; sseEvents: unknown[] | null; rawOutput?: string } {
  const rawBody = collectorData.rawResponseBody ?? null
  const sseEvents = ('sseEvents' in collectorData && Array.isArray(collectorData.sseEvents))
    ? collectorData.sseEvents
    : null
  return {
    rawBody,
    sseEvents,
    ...(rawOutput != null ? { rawOutput } : {}),
  }
}

export function createBoundModel(
  slot: ModelSlot,
  model: Model,
  connection: ModelConnection,
  driver: ExecutableDriver,
  inference: InferenceConfig = {},
  providerOptions?: ProviderOptions,
): Effect.Effect<BoundModel, never, ProviderState> {
  return Effect.map(
    ProviderState,
    (providerState) => createBoundModelImpl(slot, model, connection, driver, inference, providerOptions, providerState),
  )
}

function createBoundModelImpl(
  slot: ModelSlot,
  model: Model,
  connection: ModelConnection,
  driver: ExecutableDriver,
  inference: InferenceConfig,
  providerOptions: ProviderOptions | undefined,
  providerState: ProviderStateShape,
): BoundModel {
  const boundModel: BoundModel = {
    model,
    connection,
    invoke<I, O>(fn: ModelFunctionDef<I, O>, input: I) {
      return fn.execute(boundModel, input)
    },
    stream<K extends BamlStreamFunctionName>(functionName: K, args: readonly unknown[], options?: StreamOptions) {
      const requestInference: InferenceConfig = {
        ...inference,
        ...(options?.stopSequences ? { stopSequences: options.stopSequences } : {}),
      }

      const startMs = Date.now()
      const fallbackRequest = { input: args[0] }

      const driverRequest = {
        slot,
        functionName,
        args,
        connection,
        model,
        inference: requestInference,
        providerOptions,
      }

      let attempt = 0

      // Retry connection: invoke driver.stream(), peel first chunk to verify
      // connection is live. If first chunk fails, retry with a new connection.
      // Once first chunk succeeds, return a ChatStream whose stream starts
      // from that chunk. The peel scope is closed via Stream.ensuring when
      // the consumer finishes — no manual RELEASE, no hang.
      return Effect.retry(
        Effect.gen(function* () {
          const tracer = yield* TraceEmitter

          attempt++
          if (attempt > 1) {
            logger.warn(`[Provider] Stream connection retry attempt ${attempt}/7`)
          }

          const result = yield* driver.stream(driverRequest)
          const peelScope = yield* Scope.make()

          const [headOption, tailStream] = yield* Stream.peel(result.stream, Sink.head<string>()).pipe(
            Effect.provideService(Scope.Scope, peelScope),
            Effect.tapErrorCause(() => Scope.close(peelScope, Exit.void) as Effect.Effect<void, never, never>),
          )

          const firstChunk = Option.getOrNull(headOption)

          let traced = false
          let usageCache: ReturnType<typeof result.getUsage> | null = null
          let accumulatedOutput = ''

          const maybeTrace = () => {
            if (traced) return
            traced = true
            const usage = usageCache ?? result.getUsage()
            const collectorData = result.getCollectorData()
            Effect.runSync(
              tracer.emit({
                timestamp: new Date().toISOString(),
                model: model.id,
                provider: model.providerId,
                slot,
                request: extractTraceRequest(collectorData, fallbackRequest),
                response: extractTraceResponse(collectorData, accumulatedOutput || null),
                usage: usage ?? {
                  inputTokens: null,
                  outputTokens: null,
                  cacheReadTokens: null,
                  cacheWriteTokens: null,
                  inputCost: null,
                  outputCost: null,
                  totalCost: null,
                },
                durationMs: Date.now() - startMs,
              }),
            )
          }

          // Build the full stream: first chunk + tail, with scope cleanup, tracing, and usage accumulation
          const fullStream = (firstChunk !== null
            ? Stream.concat(Stream.make(firstChunk), tailStream)
            : tailStream
          ).pipe(
            Stream.tap((chunk) => Effect.sync(() => { accumulatedOutput += chunk })),
            Stream.ensuring(
              Effect.all([
                Scope.close(peelScope, Exit.void) as Effect.Effect<void, never, never>,
                Effect.suspend(() => {
                  const usage = result.getUsage()
                  if (usage) {
                    usageCache = usage
                    return providerState.accumulateUsage(slot, usage)
                  }
                  return Effect.void
                }),
                Effect.sync(() => maybeTrace()),
              ])
            ),
          )

          return {
            stream: fullStream,
            getUsage() {
              if (!usageCache) {
                usageCache = result.getUsage()
              }
              return usageCache
            },
            getCollectorData: result.getCollectorData,
          } satisfies ChatStream
        }),
        {
          schedule: connectionRetrySchedule,
          while: (error) => isRetryableError(error as ModelError),
        },
      )
    },
    complete<K extends BamlFunctionName>(
      functionName: K,
      args: readonly unknown[],
      _options?: CompleteOptions,
    ): Effect.Effect<CompleteResult<BamlResult<K>>, ModelError, TraceEmitter> {
      const startMs = Date.now()
      const fallbackRequest = { input: args[0] }

      return Effect.gen(function* () {
        const tracer = yield* TraceEmitter

        const { result, usage, collectorData } = yield* driver.complete({
          slot,
          functionName,
          args,
          connection,
          model,
          inference,
          providerOptions,
        })

        // Extract raw output text for tracing
        const rawOutput = typeof result === 'string' ? result
          : (result != null ? JSON.stringify(result) : null)

        yield* tracer.emit({
          timestamp: new Date().toISOString(),
          model: model.id,
          provider: model.providerId,
          slot,
          request: extractTraceRequest(collectorData, fallbackRequest),
          response: extractTraceResponse(collectorData, rawOutput),
          usage: usage ?? {
            inputTokens: null,
            outputTokens: null,
            cacheReadTokens: null,
            cacheWriteTokens: null,
            inputCost: null,
            outputCost: null,
            totalCost: null,
          },
          durationMs: Date.now() - startMs,
        })

        if (usage) yield* providerState.accumulateUsage(slot, usage)
        return { result: result as BamlResult<K>, usage }
      })
    },
  }
  return boundModel
}