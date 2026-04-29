import { Schema } from "effect"

export type ChatMessageRole = "system" | "user" | "assistant" | "tool"

export interface ChatTextContentPart {
  readonly type: "text"
  readonly text: string
}

export interface ChatImageUrlContentPart {
  readonly type: "image_url"
  readonly image_url: {
    readonly url: string
  }
}

export type ChatContentPart = ChatTextContentPart | ChatImageUrlContentPart

export interface ChatToolCall {
  readonly id: string
  readonly type: "function"
  readonly function: {
    readonly name: string
    readonly arguments: string
  }
}

export type ChatMessage =
  | {
      readonly role: "system"
      readonly content: string
    }
  | {
      readonly role: "user"
      readonly content: string | readonly ChatContentPart[]
    }
  | {
      readonly role: "assistant"
      readonly content: string | null
      readonly reasoning_content?: string | null
      readonly tool_calls?: readonly ChatToolCall[]
    }
  | {
      readonly role: "tool"
      readonly tool_call_id: string
      readonly content: string | readonly ChatContentPart[]
    }

export interface ChatTool {
  readonly type: "function"
  readonly function: {
    readonly name: string
    readonly description: string
    readonly parameters: Record<string, unknown>
  }
}

export interface ChatCompletionsRequest {
  readonly model: string
  readonly messages: readonly ChatMessage[]
  readonly tools?: readonly ChatTool[]
  readonly tool_choice?: "auto" | "none" | "required"
  readonly max_tokens?: number
  readonly stop?: readonly string[]
  readonly temperature?: number
  readonly top_p?: number
  readonly reasoning_effort?: "low" | "medium" | "high"
  readonly stream: true
  readonly stream_options?: {
    readonly include_usage: boolean
  }
}

const ChatToolCallDelta = Schema.Struct({
  index: Schema.Number,
  id: Schema.optional(Schema.NullOr(Schema.String)),
  type: Schema.optional(Schema.NullOr(Schema.Literal("function"))),
  function: Schema.optional(
    Schema.NullOr(
      Schema.Struct({
        name: Schema.optional(Schema.NullOr(Schema.String)),
        arguments: Schema.optional(Schema.NullOr(Schema.String)),
      }),
    ),
  ),
})

const ChatChunkDelta = Schema.Struct({
  role: Schema.optional(Schema.String),
  content: Schema.optional(Schema.NullOr(Schema.String)),
  reasoning_content: Schema.optional(Schema.NullOr(Schema.String)),
  tool_calls: Schema.optional(Schema.Array(ChatToolCallDelta)),
})

const ChatChunkChoice = Schema.Struct({
  index: Schema.Number,
  delta: ChatChunkDelta,
  finish_reason: Schema.optional(Schema.NullOr(Schema.String)),
})

const ChatChunkUsage = Schema.Struct({
  prompt_tokens: Schema.Number,
  completion_tokens: Schema.Number,
  prompt_tokens_details: Schema.optional(
    Schema.Struct({
      cached_tokens: Schema.optional(Schema.Number),
    }),
  ),
})

export class ChatCompletionsStreamChunk extends Schema.Class<ChatCompletionsStreamChunk>(
  "ChatCompletionsStreamChunk",
)({
  id: Schema.String,
  object: Schema.String,
  created: Schema.Number,
  model: Schema.String,
  choices: Schema.Array(ChatChunkChoice),
  usage: Schema.optional(Schema.NullOr(ChatChunkUsage)),
}) {}
