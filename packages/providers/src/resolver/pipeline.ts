import { Effect, Stream, Schedule, Duration, Scope, Exit, Option, Sink, Cause } from 'effect'
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

import { ProviderState } from '../runtime/contracts'
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
  // Some request bodies may use `input` instead of `messages`.
  if (Array.isArray(raw.input)) return { messages: raw.input }
  return fallback
}

function extractTraceResponse(
  collectorData: CollectorData,
  rawOutput: string | null,
): { rawBody: unknown | null; sseEvents: unknown[] | null; rawOutput?: string; diagnostics?: unknown | null } {
  const rawBody = collectorData.rawResponseBody ?? null
  const diagnostics = collectorData.diagnostics ?? null
  return {
    rawBody,
    sseEvents: null,
    diagnostics,
    ...(rawOutput != null ? { rawOutput } : {}),
  }
}

export function createBoundModel<TSlot extends string>(
  slot: TSlot,
  model: Model,
  connection: ModelConnection,
  driver: ExecutableDriver<TSlot>,
  inference: InferenceConfig = {},
  providerOptions?: ProviderOptions,
): Effect.Effect<BoundModel, never, ProviderState> {
  return Effect.map(
    ProviderState,
    (providerState) => createBoundModelImpl(slot, model, connection, driver, inference, providerOptions, providerState),
  )
}

function createBoundModelImpl<TSlot extends string>(
  slot: TSlot,
  model: Model,
  connection: ModelConnection,
  driver: ExecutableDriver<TSlot>,
  inference: InferenceConfig,
  providerOptions: ProviderOptions | undefined,
  providerState: import('../runtime/contracts').ProviderStateShape<string>,
): BoundModel {
  const mergeProviderOptions = (
    base: ProviderOptions | undefined,
    override: ProviderOptions | undefined,
  ): ProviderOptions | undefined => {
    if (!base) return override
    if (!override) return base
    return { ...base, ...override }
  }
  const boundModel: BoundModel = {
    model,
    connection,
    invoke<I, O>(fn: ModelFunctionDef<I, O>, input: I) {
      return fn.execute(boundModel, input)
    },
    stream<K extends BamlStreamFunctionName>(functionName: K, args: readonly unknown[], options?: StreamOptions) {
      const abortController = new AbortController()

      const requestInference: InferenceConfig = {
        ...inference,
        ...(options?.stopSequences ? { stopSequences: options.stopSequences } : {}),
      }

      const startMs = Date.now()
      const fallbackRequest = { input: args[0] }

      const streamLifecycle = {
        streamStartAtMs: Date.now(),
        firstChunkAtMs: null as number | null,
        cleanupStartAtMs: null as number | null,
        abortCalledAtMs: null as number | null,
        cleanupDoneAtMs: null as number | null,
        usageBeforeCleanup: null as ReturnType<ChatStream['getUsage']> | null,
        usageAfterCleanup: null as ReturnType<ChatStream['getUsage']> | null,
      }

      const driverRequest = {
        slot,
        functionName,
        args,
        connection,
        model,
        inference: requestInference,
        providerOptions: mergeProviderOptions(providerOptions, options?.providerOptions),
        signal: abortController.signal,
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
          if (firstChunk !== null) {
            streamLifecycle.firstChunkAtMs = Date.now()
          }

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
                slot: slot as string,
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
              Effect.catchAllCause(
                Effect.gen(function* () {
                  streamLifecycle.cleanupStartAtMs = Date.now()
                  streamLifecycle.usageBeforeCleanup = result.getUsage()
                  streamLifecycle.abortCalledAtMs = Date.now()
                  yield* Effect.sync(() => abortController.abort())
                  yield* (Scope.close(peelScope, Exit.void) as Effect.Effect<void, never, never>)
                  yield* Effect.suspend(() => {
                    const usage = result.getUsage()
                    if (usage) {
                      usageCache = usage
                      return providerState.accumulateUsage(slot, usage)
                    }
                    return Effect.void
                  })
                  streamLifecycle.usageAfterCleanup = result.getUsage()
                  streamLifecycle.cleanupDoneAtMs = Date.now()
                  yield* Effect.sync(() => maybeTrace())
                }).pipe(Effect.asVoid),
                (cause) =>
                  Effect.sync(() => {
                    logger.error({ context: 'Provider' }, `Stream cleanup failed: ${Cause.pretty(cause)}`)
                  }),
              ).pipe(Effect.asVoid),
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
            getCollectorData() {
              const collectorData = result.getCollectorData()
              if (collectorData._tag !== 'Baml') return collectorData

              return CollectorData.Baml({
                ...collectorData,
                diagnostics: collectorData.diagnostics
                  ? {
                      ...collectorData.diagnostics,
                      streamLifecycle: {
                        streamStartAtMs: streamLifecycle.streamStartAtMs,
                        firstChunkAtMs: streamLifecycle.firstChunkAtMs,
                        cleanupStartAtMs: streamLifecycle.cleanupStartAtMs,
                        abortCalledAtMs: streamLifecycle.abortCalledAtMs,
                        cleanupDoneAtMs: streamLifecycle.cleanupDoneAtMs,
                        usageBeforeCleanup: streamLifecycle.usageBeforeCleanup
                          ? {
                              inputTokens: streamLifecycle.usageBeforeCleanup.inputTokens,
                              outputTokens: streamLifecycle.usageBeforeCleanup.outputTokens,
                              cacheReadTokens: streamLifecycle.usageBeforeCleanup.cacheReadTokens,
                              cacheWriteTokens: streamLifecycle.usageBeforeCleanup.cacheWriteTokens,
                            }
                          : null,
                        usageAfterCleanup: streamLifecycle.usageAfterCleanup
                          ? {
                              inputTokens: streamLifecycle.usageAfterCleanup.inputTokens,
                              outputTokens: streamLifecycle.usageAfterCleanup.outputTokens,
                              cacheReadTokens: streamLifecycle.usageAfterCleanup.cacheReadTokens,
                              cacheWriteTokens: streamLifecycle.usageAfterCleanup.cacheWriteTokens,
                            }
                          : null,
                      },
                    }
                  : collectorData.diagnostics,
              })
            },
          } satisfies ChatStream
        }).pipe(
          Effect.onInterrupt(() => Effect.sync(() => abortController.abort())),
        ),
        {
          schedule: connectionRetrySchedule,
          while: (error) => isRetryableError(error as ModelError),
        },
      )
    },
    complete<K extends BamlFunctionName>(
      functionName: K,
      args: readonly unknown[],
      options?: CompleteOptions,
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
          providerOptions: mergeProviderOptions(providerOptions, options?.providerOptions),
        })

        // Extract raw output text for tracing
        const rawOutput = typeof result === 'string' ? result
          : (result != null ? JSON.stringify(result) : null)

        yield* tracer.emit({
          timestamp: new Date().toISOString(),
          model: model.id,
          provider: model.providerId,
          slot: slot as string,
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