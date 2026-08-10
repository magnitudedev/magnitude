import type { OptionDef, InferCallOptions } from "../options/option"
import { Option as CallOption, applyOptionDefs } from "../options/option"
import type { ChatCompletionsRequest, ChatToolChoice } from "../wire/chat-completions"
import { nativeChatCompletionsCodec } from "../codec/native-chat-completions/index"
import type { NormalizedChatCompletionsStreamChunk } from "../codec/native-chat-completions/chunk"
import {
  type ChatCompletionChunkDecoder,
  standardChatCompletionChunkDecoder,
} from "../codec/native-chat-completions/chunk-decoder"
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
  maxTokens: CallOption.define(
    (v: number) => ({ max_tokens: v }),
  ),
  temperature: CallOption.define(
    (v: number) => ({ temperature: v }),
  ),
  stop: CallOption.define(
    (v: readonly string[]) => ({ stop: [...v] }),
  ),
  topP: CallOption.define(
    (v: number) => ({ top_p: v }),
  ),
  toolChoice: CallOption.define(
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
  readonly path?: string
  readonly options: TOptions
  readonly chunkDecoder?: ChatCompletionChunkDecoder

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
    NormalizedChatCompletionsStreamChunk
  >({
    modelId: config.modelId,
    endpoint: config.endpoint,
    path: config.path ?? "/chat/completions",
    codec: nativeChatCompletionsCodec,
    doneSignal: "[DONE]",
    decodePayload: (config.chunkDecoder ?? standardChatCompletionChunkDecoder).decode,

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
