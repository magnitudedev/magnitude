import { Effect, Schema } from "effect"
import type { OptionDef, InferCallOptions } from "../options/option"
import { Option, applyOptionDefs } from "../options/option"
import type { ChatCompletionsRequest } from "../wire/chat-completions"
import { ChatCompletionsStreamChunk } from "../wire/chat-completions"
import { nativeChatCompletionsCodec } from "../codec/native-chat-completions/index"
import type { HttpConnectionFailure, StreamFailure } from "../errors/failure"
import type { ConnectionError, StreamError } from "../errors/model-error"
import { defaultClassifyConnectionError, defaultClassifyStreamError } from "../errors/classify"
import { modelDefine } from "../model/define"
import type { ModelSpec } from "../model/model-spec"

// ---------------------------------------------------------------------------
// Pre-built options for the native chat completions format
// ---------------------------------------------------------------------------

const options = {
  maxTokens: Option.define(
    (v: number) => ({ max_tokens: v }),
  ),
  temperature: Option.define(
    (v: number) => ({ temperature: v }),
  ),
  stop: Option.define(
    (v: readonly string[]) => ({ stop: [...v] }),
  ),
  topP: Option.define(
    (v: number) => ({ top_p: v }),
  ),
} as const

// ---------------------------------------------------------------------------
// NativeChatCompletions.model() config
// ---------------------------------------------------------------------------

interface NativeChatCompletionsModelConfig<
  TOptions extends Record<string, OptionDef>,
  TConnectionError = ConnectionError,
  TStreamError = StreamError,
> {
  readonly modelId: string
  readonly endpoint: string
  readonly contextWindow: number
  readonly maxOutputTokens: number
  readonly options: TOptions

  readonly compose?: (
    wire: Partial<ChatCompletionsRequest>,
    callOpts: InferCallOptions<TOptions>,
  ) => Partial<ChatCompletionsRequest>
  readonly classifyConnectionError?: (failure: HttpConnectionFailure) => TConnectionError
  readonly classifyStreamError?: (failure: StreamFailure) => TStreamError
}

// ---------------------------------------------------------------------------
// decodePayload — JSON.parse + Schema.decode for stream chunks
// ---------------------------------------------------------------------------

const decodeChatCompletionsPayload = (raw: string): Effect.Effect<ChatCompletionsStreamChunk, Error> =>
  Effect.flatMap(
    Effect.try({
      try: () => JSON.parse(raw) as unknown,
      catch: (e) => new Error(`Invalid JSON: ${raw} (${String(e)})`),
    }),
    (parsed) =>
      Effect.mapError(
        Schema.decodeUnknown(ChatCompletionsStreamChunk)(parsed),
        (e) => new Error(`Chunk decode failed: ${String(e)}`),
      ),
  )

// ---------------------------------------------------------------------------
// NativeChatCompletions.model() — overloads for default vs custom errors
// ---------------------------------------------------------------------------

/**
 * When no custom classifiers are provided, error types default to
 * ConnectionError and StreamError.
 */
function model<TOptions extends Record<string, OptionDef>>(
  config: NativeChatCompletionsModelConfig<TOptions>,
): ModelSpec<InferCallOptions<TOptions>, ConnectionError, StreamError>

/**
 * When custom classifiers are provided, error types are inferred from
 * the classifier return types.
 */
function model<
  TOptions extends Record<string, OptionDef>,
  TConnectionError,
  TStreamError,
>(
  config: NativeChatCompletionsModelConfig<TOptions, TConnectionError, TStreamError> & {
    readonly classifyConnectionError: (failure: HttpConnectionFailure) => TConnectionError
    readonly classifyStreamError: (failure: StreamFailure) => TStreamError
  },
): ModelSpec<InferCallOptions<TOptions>, TConnectionError, TStreamError>

function model<
  TOptions extends Record<string, OptionDef>,
  TConnectionError,
  TStreamError,
>(
  config: NativeChatCompletionsModelConfig<TOptions, TConnectionError, TStreamError>,
): ModelSpec<InferCallOptions<TOptions>, TConnectionError, TStreamError> {
  type TCallOptions = InferCallOptions<TOptions>

  const classifyConnection = config.classifyConnectionError
    ?? ((f: HttpConnectionFailure) => defaultClassifyConnectionError(f))
  const classifyStream = config.classifyStreamError
    ?? ((f: StreamFailure) => defaultClassifyStreamError(f))

  return modelDefine<
    TCallOptions,
    ChatCompletionsRequest,
    ChatCompletionsStreamChunk,
    TConnectionError,
    TStreamError
  >({
    modelId: config.modelId,
    endpoint: config.endpoint,
    path: "/chat/completions",
    contextWindow: config.contextWindow,
    maxOutputTokens: config.maxOutputTokens,
    codec: nativeChatCompletionsCodec,
    doneSignal: "[DONE]",
    decodePayload: decodeChatCompletionsPayload,

    classifyConnectionError: classifyConnection as (failure: HttpConnectionFailure) => TConnectionError,
    classifyStreamError: classifyStream as (failure: StreamFailure) => TStreamError,

    buildWireRequest: (prompt, tools, callOptions) => {
      // 1. Apply option defs to get wire fragments
      const optionFragments = applyOptionDefs(config.options, callOptions)

      // 2. Encode prompt via codec
      const promptFragment = nativeChatCompletionsCodec.encodePrompt(config.modelId, prompt, tools)

      // 3. Merge: protocol constants → option fragments → prompt fragment
      // Constraint-backed cast (Principle 3): the combination of protocol constants +
      // mapped option fragments + prompt fragment produces a complete ChatCompletionsRequest.
      let wire = {
        stream: true,
        stream_options: { include_usage: true },
        ...optionFragments,
        ...promptFragment,
      } as ChatCompletionsRequest

      // 4. Apply compose if provided
      if (config.compose) {
        wire = config.compose(wire, callOptions) as ChatCompletionsRequest
      }

      return wire
    },
  })
}

// ---------------------------------------------------------------------------
// NativeChatCompletions namespace
// ---------------------------------------------------------------------------

export const NativeChatCompletions = {
  model,
  options,
} as const
