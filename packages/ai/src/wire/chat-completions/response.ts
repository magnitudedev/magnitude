import { Schema } from "effect"

export const ChatToolCallDeltaSchema = Schema.Struct({
  index: Schema.Number,
  id: Schema.optionalWith(Schema.String, { as: "Option", exact: true, nullable: true }),
  type: Schema.optionalWith(Schema.Literal("function"), { as: "Option", exact: true, nullable: true }),
  function: Schema.optionalWith(
    Schema.Struct({
      name: Schema.optionalWith(Schema.String, { as: "Option", exact: true, nullable: true }),
      arguments: Schema.optionalWith(Schema.String, { as: "Option", exact: true, nullable: true }),
    }),
    { as: "Option", exact: true, nullable: true },
  ),
})

export const ChatChunkDeltaSchema = Schema.Struct({
  role: Schema.optionalWith(Schema.String, { as: "Option", exact: true, nullable: true }),
  content: Schema.optionalWith(Schema.String, { as: "Option", exact: true, nullable: true }),
  reasoning_content: Schema.optionalWith(Schema.String, { as: "Option", exact: true, nullable: true }),
  tool_calls: Schema.optionalWith(Schema.Array(ChatToolCallDeltaSchema), { as: "Option", exact: true, nullable: true }),
})

export const ChatChunkLogprobsSchema = Schema.Struct({
  content: Schema.optionalWith(Schema.Array(Schema.Struct({
    token: Schema.String,
    logprob: Schema.Number,
    top_logprobs: Schema.Array(Schema.Struct({
      token: Schema.String,
      logprob: Schema.Number,
    })),
  })), { as: "Option", exact: true, nullable: true }),
})

export const ChatChunkUsageSchema = Schema.Struct({
  prompt_tokens: Schema.Number,
  completion_tokens: Schema.Number,
  prompt_tokens_details: Schema.optionalWith(
    Schema.Struct({
      cached_tokens: Schema.optionalWith(Schema.Number, { as: "Option", exact: true, nullable: true }),
    }),
    { as: "Option", exact: true, nullable: true },
  ),
  cost: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
})

export const chatCompletionsStreamChunkFields = <A, I, R>(
  delta: Schema.Schema<A, I, R>,
) => ({
  id: Schema.String,
  object: Schema.String,
  created: Schema.Number,
  model: Schema.String,
  choices: Schema.Array(Schema.Struct({
    index: Schema.Number,
    delta,
    finish_reason: Schema.optionalWith(Schema.String, { as: "Option", exact: true, nullable: true }),
    logprobs: Schema.optionalWith(ChatChunkLogprobsSchema, { as: "Option", exact: true, nullable: true }),
  })),
  usage: Schema.optionalWith(ChatChunkUsageSchema, { as: "Option", exact: true, nullable: true }),
  raw_input: Schema.optionalWith(
    Schema.Array(
      Schema.Struct({
        text: Schema.String,
        id: Schema.Number,
      })
    ),
    { as: "Option", exact: true },
  ),
  raw_output: Schema.optionalWith(
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
    ),
    { as: "Option", exact: true },
  ),
  error: Schema.optionalWith(
    Schema.Struct({
      message: Schema.String,
      type: Schema.optionalWith(Schema.String, { as: "Option", exact: true, nullable: true }),
      code: Schema.optionalWith(Schema.String, { as: "Option", exact: true, nullable: true }),
      param: Schema.optionalWith(Schema.String, { as: "Option", exact: true, nullable: true }),
      retryable: Schema.optionalWith(Schema.Boolean, { as: "Option", exact: true, nullable: true }),
    }),
    { as: "Option", exact: true, nullable: true },
  ),
})

export class ChatCompletionsStreamChunk extends Schema.Class<ChatCompletionsStreamChunk>(
  "ChatCompletionsStreamChunk",
)(chatCompletionsStreamChunkFields(ChatChunkDeltaSchema)) {}
