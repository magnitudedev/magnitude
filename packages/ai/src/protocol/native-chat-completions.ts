import { Effect, Schema } from "effect"
import type * as ParseResult from "effect/ParseResult"
import type { JsonRecord, JsonValue } from "@magnitudedev/utils/schema"
import type { OptionDef, InferCallOptions } from "../options/option"
import { Option as CallOption, applyOptionDefs } from "../options/option"
import {
  ChatCompletionsRequestSchema,
  ChatToolChoiceSchema,
  decodeChatCompletionsRequest,
} from "../wire/chat-completions"
import { nativeChatCompletionsCodec } from "../codec/native-chat-completions/index"
import { PromptContributionError } from "../codec/native-chat-completions/encode"
import type { NormalizedChatCompletionsStreamChunk } from "../codec/native-chat-completions/chunk"
import {
  type ChatCompletionChunkDecoder,
  standardChatCompletionChunkDecoder,
} from "../codec/native-chat-completions/chunk-decoder"
import type {
  CauseInfo,
  ProviderCall,
  RejectedHttpResponse,
  StreamStartProviderCorrectnessViolation,
  StreamStartProviderRejection,
} from "../errors/failure"
import {
  causeInfoText,
  StreamStartClientCorrectnessViolation,
  toCauseInfo,
} from "../errors/failure"
import { modelDefine } from "../model/define"
import type { ModelSpec } from "../model/model-spec"
import type { ProviderModelCapabilities } from "../model/capabilities"
import type { Prompt } from "../prompt/prompt"
import type { ToolDefinition } from "../tools/tool-definition"

// ---------------------------------------------------------------------------
// Pre-built options for the native chat completions format
// ---------------------------------------------------------------------------

const options = {
  maxTokens: CallOption.field("max_tokens", Schema.Number),
  temperature: CallOption.field("temperature", Schema.Number),
  stop: CallOption.field("stop", Schema.Array(Schema.String)),
  topP: CallOption.field("top_p", Schema.Number),
  toolChoice: CallOption.field("tool_choice", ChatToolChoiceSchema),
} as const

type ChatCompletionsRequestValue = Schema.Schema.Type<typeof ChatCompletionsRequestSchema>

// ---------------------------------------------------------------------------
// NativeChatCompletions.model() config
// ---------------------------------------------------------------------------

interface NativeChatCompletionsModelConfig<
  TOptions extends Record<string, OptionDef>,
  TComposeError,
> {
  readonly modelId: string
  readonly endpoint: string
  readonly path?: string
  readonly options: TOptions
  readonly chunkDecoder?: ChatCompletionChunkDecoder

  readonly compose?: (
    request: ChatCompletionsRequestValue,
    callOptions: InferCallOptions<TOptions>,
  ) => Effect.Effect<ChatCompletionsRequestValue, TComposeError>

  readonly classifyRejectedResponse?: (
    call: ProviderCall,
    response: RejectedHttpResponse,
  ) => StreamStartProviderRejection | StreamStartProviderCorrectnessViolation
  readonly capabilities?: ProviderModelCapabilities
}

const mergeContributions = (
  call: ProviderCall,
  contributions: readonly JsonRecord[],
): Effect.Effect<JsonRecord, StreamStartClientCorrectnessViolation> => {
  const merged: Record<string, JsonValue> = {}
  for (const contribution of contributions) {
    for (const [property, value] of Object.entries(contribution)) {
      if (Object.hasOwn(merged, property)) {
        return Effect.fail(new StreamStartClientCorrectnessViolation({
          call,
          component: "request_builder",
          message: `Multiple request contributions own ${property}`,
          evidence: { _tag: "RequestContributionCollision", property },
        }))
      }
      merged[property] = value
    }
  }
  return Effect.succeed(merged)
}

const unexpectedRequestBuilderFailure = (
  call: ProviderCall,
  cause: unknown,
): StreamStartClientCorrectnessViolation => requestBuilderFailure(call, toCauseInfo(cause))

const requestBuilderFailure = (
  call: ProviderCall,
  cause: CauseInfo,
): StreamStartClientCorrectnessViolation => new StreamStartClientCorrectnessViolation({
    call,
    component: "request_builder",
    message: `Could not build native Chat Completions request: ${causeInfoText(cause)}`,
    evidence: { _tag: "UnexpectedDefectCaught", cause },
  })

const requestSchemaFailure = (
  call: ProviderCall,
  error: ParseResult.ParseError,
  subject = "Native Chat Completions request",
): StreamStartClientCorrectnessViolation => new StreamStartClientCorrectnessViolation({
  call,
  component: "request_body_encoder",
  message: `${subject} did not satisfy its Schema: ${String(error)}`,
  evidence: {
    _tag: "RequestSchemaValidationFailed",
    issue: { message: String(error) },
  },
})

const promptContributionFailure = (
  call: ProviderCall,
  error: PromptContributionError,
): StreamStartClientCorrectnessViolation => error.failure._tag === "PromptMappingFailed"
  ? requestBuilderFailure(call, error.failure.cause)
  : requestSchemaFailure(call, error.failure.error, "Native Chat Completions prompt contribution")

interface NativeChatCompletionsRequestConfig<
  TOptions extends Record<string, OptionDef>,
  TComposeError,
> {
  readonly call: ProviderCall
  readonly modelId: string
  readonly options: TOptions
  readonly compose?: (
    request: ChatCompletionsRequestValue,
    callOptions: InferCallOptions<TOptions>,
  ) => Effect.Effect<ChatCompletionsRequestValue, TComposeError>
}

const buildRequest = <
  TOptions extends Record<string, OptionDef>,
  TComposeError,
>(
  config: NativeChatCompletionsRequestConfig<TOptions, TComposeError>,
  prompt: Prompt,
  tools: readonly ToolDefinition[],
  callOptions: InferCallOptions<TOptions>,
): Effect.Effect<ChatCompletionsRequestValue, StreamStartClientCorrectnessViolation> =>
  Effect.gen(function* () {
    const optionFragments = yield* applyOptionDefs(config.options, callOptions).pipe(
      Effect.mapError((error) => error.failure._tag === "OptionMappingFailed"
        ? requestBuilderFailure(config.call, error.failure.cause)
        : requestSchemaFailure(
            config.call,
            error.failure.error,
            `Call option ${error.option} contribution`,
          )),
    )
    const promptFragment = yield* nativeChatCompletionsCodec
      .encodePrompt(config.modelId, prompt, tools)
      .pipe(Effect.mapError((error) => promptContributionFailure(config.call, error)))
    const protocol = { stream: true, stream_options: { include_usage: true } } as const
    const draft = yield* mergeContributions(config.call, [protocol, ...optionFragments, promptFragment])
    const request = yield* decodeChatCompletionsRequest(draft).pipe(
      Effect.mapError((error) => requestSchemaFailure(config.call, error)),
    )
    const compose = config.compose
    if (compose === undefined) return request

    const composeEffect = yield* Effect.try({
      try: () => compose(request, callOptions),
      catch: (cause) => unexpectedRequestBuilderFailure(config.call, cause),
    })
    return yield* composeEffect.pipe(
      Effect.mapError((cause) => unexpectedRequestBuilderFailure(config.call, cause)),
    )
  })

// ---------------------------------------------------------------------------
// NativeChatCompletions.model()
// ---------------------------------------------------------------------------

function model<
  TOptions extends Record<string, OptionDef>,
  TComposeError = never,
>(
  config: NativeChatCompletionsModelConfig<TOptions, TComposeError>,
): ModelSpec<InferCallOptions<TOptions>> {
  type TCallOptions = InferCallOptions<TOptions>

  return modelDefine({
    modelId: config.modelId,
    endpoint: config.endpoint,
    path: config.path ?? "/chat/completions",
    codec: nativeChatCompletionsCodec,
    requestSchema: ChatCompletionsRequestSchema,
    doneSignal: "[DONE]",
    decodePayload: (config.chunkDecoder ?? standardChatCompletionChunkDecoder).decode,

    classifyRejectedResponse: config.classifyRejectedResponse,
    capabilities: config.capabilities,

    buildRequest: (call, prompt, tools, callOptions: TCallOptions) => buildRequest(
      {
        call,
        modelId: config.modelId,
        options: config.options,
        ...(config.compose === undefined ? {} : { compose: config.compose }),
      },
      prompt,
      tools,
      callOptions,
    ),
  })
}

// ---------------------------------------------------------------------------
// NativeChatCompletions namespace
// ---------------------------------------------------------------------------

export const NativeChatCompletions = {
  buildRequest,
  model,
  options,
} as const
