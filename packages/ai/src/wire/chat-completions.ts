import { Schema } from "effect"
import type { JsonSchemaObject } from "@magnitudedev/utils/schema"

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

export interface ChatNamedFunctionToolChoice {
  readonly type: "function"
  readonly function: {
    readonly name: string
  }
}

export interface ChatAllowedToolsToolChoice {
  readonly type: "allowed_tools"
  readonly allowed_tools: {
    readonly mode: "auto" | "required"
    readonly tools: ReadonlyArray<{
      readonly type: "function"
      readonly function: { readonly name: string }
    }>
  }
}

export type ChatToolChoice =
  | "auto"
  | "none"
  | "required"
  | ChatNamedFunctionToolChoice
  | ChatAllowedToolsToolChoice

export interface ChatTool {
  readonly type: "function"
  readonly function: {
    readonly name: string
    readonly description: string
    readonly parameters: JsonSchemaObject
  }
}

export interface ChatCompletionsRequest {
  readonly model: string
  readonly messages: readonly ChatMessage[]
  readonly tools?: readonly ChatTool[]
  readonly tool_choice?: ChatToolChoice
  readonly max_tokens?: number
  readonly stop?: readonly string[]
  readonly temperature?: number
  readonly top_p?: number
  readonly reasoning_effort?: "none" | "low" | "medium" | "high"
  readonly logprobs?: boolean
  readonly top_logprobs?: number
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
  role: Schema.optional(Schema.NullOr(Schema.String)),
  content: Schema.optional(Schema.NullOr(Schema.String)),
  reasoning_content: Schema.optional(Schema.NullOr(Schema.String)),
  tool_calls: Schema.optional(Schema.NullOr(Schema.Array(ChatToolCallDelta))),
})

const ChatChunkLogprobs = Schema.Struct({
  content: Schema.optional(Schema.NullOr(Schema.Array(Schema.Struct({
    token: Schema.String,
    logprob: Schema.Number,
    top_logprobs: Schema.Array(Schema.Struct({
      token: Schema.String,
      logprob: Schema.Number,
    })),
  })))),
})

const ChatChunkChoice = Schema.Struct({
  index: Schema.Number,
  delta: ChatChunkDelta,
  finish_reason: Schema.optional(Schema.NullOr(Schema.String)),
  logprobs: Schema.optional(Schema.NullOr(ChatChunkLogprobs)),
})

const ChatChunkUsage = Schema.Struct({
  prompt_tokens: Schema.Number,
  completion_tokens: Schema.Number,
  prompt_tokens_details: Schema.optional(
    Schema.NullOr(
      Schema.Struct({
        cached_tokens: Schema.optional(Schema.NullOr(Schema.Number)),
      }),
    ),
  ),
  cost: Schema.optional(Schema.Number),
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
  raw_input: Schema.optional(
    Schema.Array(
      Schema.Struct({
        text: Schema.String,
        id: Schema.Number,
      })
    )
  ),
  raw_output: Schema.optional(
    Schema.Array(
      Schema.Struct({
        text: Schema.String,
        id: Schema.Number,
        logprobs: Schema.NullOr(
          Schema.Array(
            Schema.Struct({
              text: Schema.String,
              logprob: Schema.Number,
            })
          )
        ),
      })
    )
  ),
  error: Schema.optional(
    Schema.NullOr(
      Schema.Struct({
        message: Schema.String,
        type: Schema.optional(Schema.NullOr(Schema.String)),
        code: Schema.optional(Schema.NullOr(Schema.String)),
        param: Schema.optional(Schema.NullOr(Schema.String)),
      }),
    ),
  ),
}) {}
