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
    } | null
  }
  Responses: {
    readonly rawRequestBody: unknown
    readonly rawResponseBody: unknown
    readonly sseEvents: unknown[] | null
    readonly diagnostics?: {
      readonly codexVariant: 'openai-codex' | 'copilot-codex' | null
      readonly providerId: string | null
      readonly modelId: string | null
      readonly authType: string | null
      readonly driverId: 'openai-responses'
      readonly endpoint: string | null
      readonly terminalEventType: string | null
      readonly terminalEventPayload: unknown | null
      readonly usageSource:
        | 'response.completed.response.usage'
        | 'response.completed.usage'
        | 'response.other'
        | 'raw-response-body.usage'
        | 'fallback-retrieve.response.usage'
        | 'fallback-retrieve.usage'
        | 'none'
      readonly usagePath:
        | 'event.response.usage'
        | 'event.usage'
        | 'response.completed.response.usage'
        | 'response.completed.usage'
        | 'rawResponseBody.usage'
        | 'fallbackRetrieve.response.usage'
        | 'fallbackRetrieve.usage'
        | 'none'
        | null
      readonly rawUsage: unknown | null
      readonly parsedInputTokens: number | null
      readonly parsedOutputTokens: number | null
      readonly parsedCacheReadTokens: number | null
      readonly parsedCacheWriteTokens: number | null
      readonly selectedUsageEventType: string | null
      readonly usageRejectionReasons: readonly string[]
      readonly usageAbsent: boolean
      readonly streamEndReason: 'done-sentinel' | 'eof' | 'aborted' | 'error' | 'unknown'
      readonly sawDoneSentinel: boolean
      readonly sawEof: boolean
      readonly sawAbort: boolean
      readonly streamError: string | null
      readonly parsedEventCount: number
      readonly usageBearingEventCount: number
      readonly eventTypeCounts: Readonly<Record<string, number>>
      readonly terminalCompletedCount: number
      readonly terminalIncompleteCount: number
      readonly terminalFailedCount: number
      readonly responseIdSeen: boolean
      readonly responseId: string | null
      readonly fallbackRetrieveUsed: boolean
      readonly fallbackRetrieveSucceeded: boolean
      readonly fallbackRetrieveUsageFound: boolean
      readonly fallbackRetrieveUsagePath:
        | 'fallbackRetrieve.response.usage'
        | 'fallbackRetrieve.usage'
        | null
      readonly rawStreamTail: readonly string[]
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