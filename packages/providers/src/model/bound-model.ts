import type { Effect, Stream } from 'effect'
import type { Model } from './model'
import type { ModelConnection } from './model-connection'
import type { CallUsage } from '../state/provider-state'
import type { CollectorData } from '../drivers/types'
import type { TraceEmitter } from '../resolver/tracing'
import type { BamlFunctionName, BamlResult, BamlStreamFunctionName } from '../drivers/baml-types'
import type { ModelError } from '../errors/model-error'

export interface StreamOptions {
  readonly stopSequences?: string[]
}

export interface CompleteOptions {}

export interface ChatStream {
  readonly stream: Stream.Stream<string, ModelError>
  getUsage(): CallUsage
  getCollectorData(): CollectorData
}

export interface CompleteResult<T = unknown> {
  readonly result: T
  readonly usage: CallUsage
}

export interface StreamingFn<I, O> {
  readonly name: string
  readonly mode: 'stream'
  readonly execute: (model: BoundModel, input: I) => Effect.Effect<O, ModelError, TraceEmitter>
}

export interface CompleteFn<I, O> {
  readonly name: string
  readonly mode: 'complete'
  readonly execute: (model: BoundModel, input: I) => Effect.Effect<O, ModelError, TraceEmitter>
}

export type ModelFunctionDef<I, O> = StreamingFn<I, O> | CompleteFn<I, O>

export interface BoundModel {
  readonly model: Model
  readonly connection: ModelConnection

  /** Typed invoke for streaming functions */
  invoke<I, O>(fn: StreamingFn<I, O>, input: I): Effect.Effect<O, ModelError, TraceEmitter>
  /** Typed invoke for complete functions */
  invoke<I, O>(fn: CompleteFn<I, O>, input: I): Effect.Effect<O, ModelError, TraceEmitter>

  /** @internal Raw stream — prefer invoke() */
  stream<K extends BamlStreamFunctionName>(
    functionName: K,
    args: readonly unknown[],
    options?: StreamOptions,
  ): Effect.Effect<ChatStream, ModelError, TraceEmitter>

  /** @internal Raw complete — prefer invoke() */
  complete<K extends BamlFunctionName>(
    functionName: K,
    args: readonly unknown[],
    options?: CompleteOptions,
  ): Effect.Effect<CompleteResult<BamlResult<K>>, ModelError, TraceEmitter>
}