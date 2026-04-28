import { Schema } from 'effect'

// =============================================================================
// Request types (plain interfaces — not validated; producers own correctness)
// =============================================================================

export type ChatMessageRole = 'system' | 'user' | 'assistant' | 'tool'

/** A single function tool-call attached to an assistant message. */
export interface ChatToolCall {
  readonly id:       string
  readonly type:     'function'
  readonly function: {
    readonly name:      string
    readonly arguments: string  // JSON string
  }
}

/**
 * ChatMessage — the wire-level messages array entry.
 * Discriminated by role; each variant carries the fields that role allows.
 */
export type ChatMessage =
  | {
      readonly role:    'system'
      readonly content: string
    }
  | {
      readonly role:    'user'
      /**
       * Content may be a plain string or an array of content parts (for
       * multimodal input). Keep as unknown so callers can pass either
       * without a cast; the provider accepts both.
       */
      readonly content: string | readonly unknown[]
    }
  | {
      readonly role:           'assistant'
      readonly content?:       string | null
      /** Native reasoning content (Kimi K2, DeepSeek R1, etc.) */
      readonly reasoning_content?: string | null
      readonly tool_calls?:    readonly ChatToolCall[]
    }
  | {
      readonly role:        'tool'
      readonly tool_call_id: string
      /**
       * Tool result content. May be a plain string or an array of content
       * parts (e.g. to embed images). Kept as unknown for the same reason
       * as user content above.
       */
      readonly content:     string | readonly unknown[]
    }

/** A single tool declaration in the request tools array. */
export interface ChatTool {
  readonly type:     'function'
  readonly function: {
    readonly name:        string
    readonly description: string
    readonly parameters:  unknown  // JSON Schema object
  }
}

/** Full chat completions request object. */
export interface ChatCompletionsRequest {
  readonly model:             string
  readonly messages:          readonly ChatMessage[]
  readonly tools?:            readonly ChatTool[]
  readonly tool_choice?:      'auto' | 'none' | 'required'
  readonly max_tokens?:       number
  readonly temperature?:      number
  readonly stop?:             readonly string[]
  readonly stream:            true
  readonly stream_options?:   { readonly include_usage: boolean }
}

// =============================================================================
// Stream chunk types (Schema.Class — these are wire-boundary values we decode)
// =============================================================================

/**
 * A tool-call delta fragment inside a stream chunk choice.
 * index is the parallel tool-call slot; id/name only appear in the first
 * fragment for each slot.
 */
const ToolCallDelta = Schema.Struct({
  index:    Schema.Number,
  id:       Schema.optional(Schema.NullOr(Schema.String)),
  type:     Schema.optional(Schema.NullOr(Schema.Literal('function'))),
  function: Schema.optional(Schema.NullOr(Schema.Struct({
    name:      Schema.optional(Schema.NullOr(Schema.String)),
    arguments: Schema.optional(Schema.NullOr(Schema.String)),
  }))),
})

/** A choice delta — the incremental content emitted per chunk. */
const ChunkDelta = Schema.Struct({
  role:              Schema.optional(Schema.String),
  content:           Schema.optional(Schema.NullOr(Schema.String)),
  /** Reasoning / thinking content (Kimi K2.6, DeepSeek R1, QwQ). */
  reasoning_content: Schema.optional(Schema.NullOr(Schema.String)),
  tool_calls:        Schema.optional(Schema.Array(ToolCallDelta)),
})

/** A single choice entry in the chunk. */
const ChunkChoice = Schema.Struct({
  index:         Schema.Number,
  delta:         ChunkDelta,
  finish_reason: Schema.optional(Schema.NullOr(Schema.String)),
})

/** Token usage — present in the final chunk when stream_options.include_usage is true. */
const ChunkUsage = Schema.Struct({
  prompt_tokens:     Schema.Number,
  completion_tokens: Schema.Number,
  prompt_tokens_details: Schema.optional(Schema.Struct({
    cached_tokens: Schema.optional(Schema.Number),
  })),
})

/**
 * ChatCompletionsStreamChunk — the decoded SSE data payload.
 * This is a Schema.Class so the driver can call Schema.decodeUnknown at
 * the wire boundary to catch malformed payloads early.
 */
export class ChatCompletionsStreamChunk extends Schema.Class<ChatCompletionsStreamChunk>(
  'ChatCompletionsStreamChunk',
)({
  id:      Schema.String,
  object:  Schema.String,
  created: Schema.Number,
  model:   Schema.String,
  choices: Schema.Array(ChunkChoice),
  usage:   Schema.optional(Schema.NullOr(ChunkUsage)),
}) {}
