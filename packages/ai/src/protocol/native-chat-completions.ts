import { Data, Effect, Schema } from "effect"
import type { OptionDef, InferCallOptions } from "../options/option"
import { Option, applyOptionDefs } from "../options/option"
import type { ChatCompletionsRequest, ChatToolChoice } from "../wire/chat-completions"
import { ChatCompletionsStreamChunk } from "../wire/chat-completions"
import { nativeChatCompletionsCodec } from "../codec/native-chat-completions/index"
import type {
  ProviderCall,
  RejectedHttpResponse,
  StreamStartProviderCorrectnessViolation,
  StreamStartProviderRejection,
} from "../errors/failure"
import { modelDefine } from "../model/define"
import type { ModelSpec } from "../model/model-spec"
import type { ProviderModelCapabilities } from "../model/capabilities"

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
  toolChoice: Option.define(
    (v: ChatToolChoice) => ({ tool_choice: v }),
  ),
} as const

// ---------------------------------------------------------------------------
// NativeChatCompletions.model() config
// ---------------------------------------------------------------------------

interface NativeChatCompletionsModelConfig<
  TOptions extends Record<string, OptionDef>,
> {
  readonly modelId: string
  readonly endpoint: string
  readonly options: TOptions

  readonly compose?: (
    wire: Partial<ChatCompletionsRequest>,
    callOpts: InferCallOptions<TOptions>,
  ) => Partial<ChatCompletionsRequest>
  readonly classifyRejectedResponse?: (
    call: ProviderCall,
    response: RejectedHttpResponse,
  ) => StreamStartProviderRejection | StreamStartProviderCorrectnessViolation
  readonly capabilities?: ProviderModelCapabilities
}

// ---------------------------------------------------------------------------
// decodePayload — JSON.parse + Schema.decode for stream chunks
// ---------------------------------------------------------------------------

class ChatPayloadJsonParseError extends Data.TaggedError("ChatPayloadJsonParseError")<{
  readonly message: string
  readonly raw: string
  readonly cause: unknown
}> {}

class ChatPayloadSchemaDecodeError extends Data.TaggedError("ChatPayloadSchemaDecodeError")<{
  readonly message: string
  readonly raw: string
  readonly cause: unknown
}> {}

const decodeChatCompletionsPayload = (raw: string): Effect.Effect<
  ChatCompletionsStreamChunk,
  ChatPayloadJsonParseError | ChatPayloadSchemaDecodeError
> =>
  Effect.flatMap(
    Effect.try({
      try: () => JSON.parse(raw) as unknown,
      catch: (cause) => new ChatPayloadJsonParseError({
        message: `Invalid JSON: ${raw} (${String(cause)})`,
        raw,
        cause,
      }),
    }),
    (parsed) =>
      Effect.mapError(
        Schema.decodeUnknown(ChatCompletionsStreamChunk)(parsed),
        (cause) => new ChatPayloadSchemaDecodeError({
          message: `Chunk decode failed: ${String(cause)}`,
          raw,
          cause,
        }),
      ),
  )

// ---------------------------------------------------------------------------
// NativeChatCompletions.model()
// ---------------------------------------------------------------------------

function model<
  TOptions extends Record<string, OptionDef>,
>(
  config: NativeChatCompletionsModelConfig<TOptions>,
): ModelSpec<InferCallOptions<TOptions>> {
  type TCallOptions = InferCallOptions<TOptions>

  return modelDefine<
    TCallOptions,
    ChatCompletionsRequest,
    ChatCompletionsStreamChunk
  >({
    modelId: config.modelId,
    endpoint: config.endpoint,
    path: "/chat/completions",
    codec: nativeChatCompletionsCodec,
    doneSignal: "[DONE]",
    decodePayload: decodeChatCompletionsPayload,

    classifyRejectedResponse: config.classifyRejectedResponse,
    capabilities: config.capabilities,

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

      // 4. Default tool_choice to "auto" when tools are present and no explicit choice
      if (wire.tools && wire.tools.length > 0 && !wire.tool_choice) {
        wire = { ...wire, tool_choice: "auto" }
      }

      // 5. Apply compose if provided
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
