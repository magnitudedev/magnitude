import type { Stream } from "effect"
import type { Prompt, PromptShape } from "../prompt/prompt"
import type { ToolDefinition } from "../tools/tool-definition"
import type { ModelError } from "../errors/model-error"
import type { ResponseStreamEvent } from "../response/events"
import type {
  ChatCompletionsRequest,
  ChatCompletionsStreamChunk,
} from "../wire/chat-completions"

export interface EncodeOptions {
  readonly maxTokens?: number
  readonly stop?: readonly string[]
  readonly temperature?: number
  readonly topP?: number
  readonly reasoningEffort?: "low" | "medium" | "high"
}

export interface Codec {
  readonly id: string
  readonly encode: (
    model: string,
    prompt: Prompt | PromptShape,
    tools: readonly ToolDefinition<any, any>[],
    options: EncodeOptions,
  ) => ChatCompletionsRequest
  readonly decode: (
    chunks: Stream.Stream<ChatCompletionsStreamChunk, ModelError>,
  ) => Stream.Stream<ResponseStreamEvent, ModelError>
}
