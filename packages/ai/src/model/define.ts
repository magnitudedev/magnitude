import { Effect, Stream } from "effect"
import type * as HttpClient from "@effect/platform/HttpClient"
import { Prompt } from "../prompt/prompt"
import type { ToolDefinition } from "../tools/tool-definition"
import type { ResponseStreamEvent } from "../response/events"
import type { AuthApplicator } from "../auth/auth"
import type { Codec } from "../codec/codec"
import type { HttpConnectionFailure, StreamFailure } from "../errors/failure"
import { executeHttpStream } from "../transport/stream"
import type { ModelSpec } from "./model-spec"
import type { BoundModel } from "./bound-model"

// ---------------------------------------------------------------------------
// Model.define — internal factory used by protocol namespaces
// ---------------------------------------------------------------------------

export interface ModelDefineConfig<
  TCallOptions,
  TWireReq,
  TWireChunk,
  TConnectionError,
  TStreamError,
> {
  readonly modelId: string
  readonly endpoint: string
  readonly path: string
  readonly contextWindow: number
  readonly maxOutputTokens: number
  readonly codec: Codec<TWireReq, TWireChunk>
  readonly buildWireRequest: (
    prompt: Prompt,
    tools: readonly ToolDefinition[],
    options: TCallOptions,
  ) => TWireReq
  readonly classifyConnectionError: (failure: HttpConnectionFailure) => TConnectionError
  readonly classifyStreamError: (failure: StreamFailure) => TStreamError
  readonly decodePayload: (raw: string) => Effect.Effect<TWireChunk, Error>
  readonly doneSignal?: string
}

function joinUrl(endpoint: string, path: string): string {
  return endpoint.replace(/\/+$/, "") + path
}

export function modelDefine<
  TCallOptions,
  TWireReq,
  TWireChunk,
  TConnectionError,
  TStreamError,
>(
  config: ModelDefineConfig<TCallOptions, TWireReq, TWireChunk, TConnectionError, TStreamError>,
): ModelSpec<TCallOptions, TConnectionError, TStreamError> {
  const url = joinUrl(config.endpoint, config.path)

  const spec: ModelSpec<TCallOptions, TConnectionError, TStreamError> = {
    modelId: config.modelId,
    endpoint: config.endpoint,
    contextWindow: config.contextWindow,
    maxOutputTokens: config.maxOutputTokens,

    bind: (args) => modelBind(spec, args.auth, args.defaults),

    _execute: (
      auth: AuthApplicator,
      prompt: Prompt,
      tools: readonly ToolDefinition[],
      options: TCallOptions,
    ): Effect.Effect<
      Stream.Stream<ResponseStreamEvent, TStreamError>,
      TConnectionError,
      HttpClient.HttpClient
    > => {
      const wireRequest = config.buildWireRequest(prompt, tools, options)

      const httpEffect = executeHttpStream({
        url,
        body: wireRequest,
        auth,
        decodePayload: config.decodePayload,
        doneSignal: config.doneSignal,
      })

      return httpEffect.pipe(
        Effect.map((wireStream) =>
          config.codec.decode(wireStream.pipe(Stream.mapError(config.classifyStreamError))),
        ),
        Effect.mapError(config.classifyConnectionError),
      )
    },
  }

  return spec
}

// ---------------------------------------------------------------------------
// Model.bind — public binding API
// ---------------------------------------------------------------------------

export function modelBind<
  TCallOptions,
  TConnectionError,
  TStreamError,
>(
  spec: ModelSpec<TCallOptions, TConnectionError, TStreamError>,
  auth: AuthApplicator,
  defaults?: Partial<TCallOptions>,
): BoundModel<TCallOptions, TConnectionError, TStreamError> {
  return {
    spec,
    stream: (prompt, tools, options?) => {
      const merged = { ...defaults, ...options } as TCallOptions
      return spec._execute(auth, prompt, tools, merged)
    },
  }
}

// ---------------------------------------------------------------------------
// Model namespace — public API
// ---------------------------------------------------------------------------

export const Model = {
  /** @internal — used by protocol namespaces */
  define: modelDefine,
  /** Bind a ModelSpec with auth and optional defaults to create a BoundModel */
  bind: modelBind,
} as const
