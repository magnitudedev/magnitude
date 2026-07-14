import {
  NativeChatCompletions,
  Option,
  type BaseCallOptions,
  type BoundModel,
  type ChatCompletionsRequest,
  type ModelSpec,
  type ProviderCall,
  type RejectedHttpResponse,
  type StreamStartProviderCorrectnessViolation,
  type StreamStartProviderRejection,
  type ToolChoice,
} from "@magnitudedev/ai"
import type { ChatToolChoice } from "@magnitudedev/ai"
import { classifyOpenAiCompatibleRejectedResponse } from "./errors"

export interface OpenAiCompatibleCallOptions {
  readonly maxTokens?: number
  readonly toolChoice?: ChatToolChoice
  readonly reasoningEffort?: string
}

export interface OpenAiCompatibleSpecConfig {
  readonly modelId: string
  readonly endpoint: string
  readonly providerName: string
  readonly reasoningField?: "reasoning" | "reasoning_content"
  readonly preserveReasoningDetails?: boolean
  readonly reasoningRequestMode?: ReasoningRequestMode
  readonly classifyRejectedResponse?: (
    call: ProviderCall,
    response: RejectedHttpResponse,
  ) => StreamStartProviderRejection | StreamStartProviderCorrectnessViolation
}

export type ReasoningRequestMode =
  | "openai"
  | "openrouter"
  | "thinking-toggle"
  | "thinking-effort"
  | "kimi"

const options = {
  maxTokens: NativeChatCompletions.options.maxTokens,
  toolChoice: Option.define((value: ChatToolChoice) => ({ tool_choice: value })),
  reasoningEffort: Option.define((value: string) => ({ reasoning_effort: value })),
} as const

type ExtendedChatCompletionsRequest = Partial<ChatCompletionsRequest> & {
  readonly reasoning?: { readonly effort: string }
  readonly thinking?: {
    readonly type: "enabled" | "disabled"
    readonly effort?: string
    readonly keep?: string
  }
}

export function composeReasoningRequest(
  wire: Partial<ChatCompletionsRequest>,
  mode: ReasoningRequestMode,
): ExtendedChatCompletionsRequest {
  const effort = wire.reasoning_effort as string | undefined
  if (effort === undefined) return wire

  const { reasoning_effort: _reasoningEffort, ...rest } = wire
  if (effort === "default") return rest
  if (mode === "openai") return { ...rest, reasoning_effort: effort } as ExtendedChatCompletionsRequest
  if (mode === "openrouter") return { ...rest, reasoning: { effort } }
  if (effort === "none") return { ...rest, thinking: { type: "disabled" } }
  if (mode === "thinking-toggle") return { ...rest, thinking: { type: "enabled" } }
  if (mode === "kimi") {
    return { ...rest, thinking: { type: "enabled", effort, keep: "all" } }
  }
  return {
    ...rest,
    reasoning_effort: effort,
    thinking: { type: "enabled" },
  } as ExtendedChatCompletionsRequest
}

export function createOpenAiCompatibleSpec(
  config: OpenAiCompatibleSpecConfig,
): ModelSpec<OpenAiCompatibleCallOptions> {
  return NativeChatCompletions.model({
    modelId: config.modelId,
    endpoint: config.endpoint,
    options,
    compose: (wire) => {
      const withReasoning = composeReasoningRequest(
        wire,
        config.reasoningRequestMode ?? "openai",
      )
      const withReasoningField = config.reasoningField === "reasoning"
        ? {
            ...withReasoning,
            messages: withReasoning.messages?.map((message) => {
              if (message.role !== "assistant" || message.reasoning_content === undefined) {
                return message
              }
              const { reasoning_content, ...rest } = message
              return { ...rest, reasoning: reasoning_content }
            }),
          }
        : withReasoning
      if (config.preserveReasoningDetails) return withReasoningField
      return {
        ...withReasoningField,
        messages: withReasoningField.messages?.map((message) => {
          if (message.role !== "assistant" || message.reasoning_details === undefined) return message
          const { reasoning_details: _reasoningDetails, ...rest } = message
          return rest
        }),
      }
    },
    classifyRejectedResponse: config.classifyRejectedResponse
      ?? ((call, response) => classifyOpenAiCompatibleRejectedResponse(config.providerName, call, response)),
  })
}

function normalizeToolChoice(choice: ToolChoice | undefined): ChatToolChoice | undefined {
  if (!choice) return undefined
  if (typeof choice === "string") return choice
  if (choice.type === "grammar") return "required"
  if (choice.type === "allowed_tools") {
    const tools = choice.allowed_tools.tools
    if (tools.length === 1) {
      return { type: "function", function: { name: tools[0].function.name } }
    }
    return choice.allowed_tools.mode
  }
  return choice
}

export function wrapOpenAiCompatibleAsBaseModel(
  internal: BoundModel<OpenAiCompatibleCallOptions>,
): BoundModel<BaseCallOptions> {
  return {
    stream: (prompt, tools, callOptions) => internal.stream(prompt, tools, {
      ...(callOptions?.maxTokens !== undefined ? { maxTokens: callOptions.maxTokens } : {}),
      ...(callOptions?.reasoningEffort !== undefined ? { reasoningEffort: callOptions.reasoningEffort } : {}),
      ...(callOptions?.toolChoice !== undefined
        ? { toolChoice: normalizeToolChoice(callOptions.toolChoice) }
        : {}),
      ...(callOptions?.generateToolCallId ? { generateToolCallId: callOptions.generateToolCallId } : {}),
    }),
  }
}
