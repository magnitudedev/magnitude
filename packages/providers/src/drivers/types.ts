import { Data, type Effect, type Stream } from 'effect'
import type { Model } from '../model/model'
import type { ModelConnection } from '../model/model-connection'
import type { InferenceConfig } from '../model/inference-config'
import type { CallUsage } from '../state/provider-state'
import type { ModelError } from '../errors/model-error'
import type { ModelDriverId } from '../model/model-driver'
import type { AuthInfo, ProviderOptions } from '../types'
import type { ModelSlot } from '../state/provider-state'

export interface DriverRequest {
  readonly slot: ModelSlot
  readonly functionName: string
  readonly args: readonly unknown[]
  readonly connection: ModelConnection
  readonly model: Model
  readonly inference: InferenceConfig
  readonly providerOptions?: ProviderOptions
}

export type CollectorData = Data.TaggedEnum<{
  Baml: { readonly rawRequestBody: unknown; readonly rawResponseBody: unknown }
  Responses: { readonly rawRequestBody: unknown; readonly rawResponseBody: unknown; readonly sseEvents: unknown[] | null }
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

export interface ExecutableDriver {
  readonly id: ModelDriverId
  connect(model: Model, auth: AuthInfo | null, inference: InferenceConfig): Effect.Effect<ModelConnection, ModelError>
  stream(req: DriverRequest): Effect.Effect<StreamResult, ModelError>
  complete<T = unknown>(req: DriverRequest): Effect.Effect<CompleteResult<T>, ModelError>
}