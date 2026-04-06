import { Data, type Effect, type Stream } from 'effect'
import type { Model } from '../model/model'
import type { ModelConnection } from '../model/model-connection'
import type { InferenceConfig } from '../model/inference-config'
import type { CallUsage } from '../state/provider-state'
import type { ModelError } from '../errors/model-error'
import type { ModelDriverId } from '../model/model-driver'
import type { AuthInfo, ProviderOptions } from '../types'
export interface DriverRequest<TSlot extends string = string> {
  readonly slot: TSlot
  readonly functionName: string
  readonly args: readonly unknown[]
  readonly connection: ModelConnection
  readonly model: Model
  readonly inference: InferenceConfig
  readonly providerOptions?: ProviderOptions
  readonly signal?: AbortSignal
}

export type CollectorData = Data.TaggedEnum<{
  Baml: {
    readonly rawRequestBody: unknown
    readonly rawResponseBody: unknown
    readonly diagnostics?: {
      readonly usageSource: 'http-response-usage' | 'anthropic-sse-usage' | 'collector-usage' | 'none'
      readonly rawUsage: unknown | null
      readonly parsedInputTokens: number | null
      readonly parsedOutputTokens: number | null
      readonly parsedCacheReadTokens: number | null
      readonly parsedCacheWriteTokens: number | null
      readonly providerId: string | null
      readonly modelId: string | null
      readonly authType: string | null
      readonly driverId: 'baml'
      readonly usageAbsent: boolean
      readonly streamLifecycle?: {
        readonly streamStartAtMs: number
        readonly firstChunkAtMs: number | null
        readonly cleanupStartAtMs: number | null
        readonly abortCalledAtMs: number | null
        readonly cleanupDoneAtMs: number | null
        readonly usageBeforeCleanup: {
          readonly inputTokens: number | null
          readonly outputTokens: number | null
          readonly cacheReadTokens: number | null
          readonly cacheWriteTokens: number | null
        } | null
        readonly usageAfterCleanup: {
          readonly inputTokens: number | null
          readonly outputTokens: number | null
          readonly cacheReadTokens: number | null
          readonly cacheWriteTokens: number | null
        } | null
      } | null
    } | null
  }
}>

export const CollectorData = Data.taggedEnum<CollectorData>()

export interface StreamResult {
  readonly stream: Stream.Stream<string, ModelError>
  getUsage(): CallUsage
  getCollectorData(): CollectorData
}

export interface CompleteResult<T = unknown> {
  readonly result: T
  readonly usage: CallUsage
  readonly collectorData: CollectorData
}

export interface ExecutableDriver<TSlot extends string = string> {
  readonly id: ModelDriverId
  connect(model: Model, auth: AuthInfo | null, inference: InferenceConfig): Effect.Effect<ModelConnection, ModelError>
  stream(req: DriverRequest<TSlot>): Effect.Effect<StreamResult, ModelError>
  complete<T = unknown>(req: DriverRequest<TSlot>): Effect.Effect<CompleteResult<T>, ModelError>
}