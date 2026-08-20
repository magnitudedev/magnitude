import { Effect, Schema } from "effect"
import type * as ParseResult from "effect/ParseResult"
import {
  JsonRecordSchema,
  type JsonSchemaObject,
  type JsonValue,
} from "@magnitudedev/utils/schema"

const ChatTextContentPartSchema = Schema.Struct({ type: Schema.Literal("text"), text: Schema.String })
const ChatImageUrlContentPartSchema = Schema.Struct({
  type: Schema.Literal("image_url"),
  image_url: Schema.Struct({ url: Schema.String }),
})
const ChatContentPartSchema = Schema.Union(ChatTextContentPartSchema, ChatImageUrlContentPartSchema)

const ChatToolCallSchema = Schema.Struct({
  id: Schema.String,
  type: Schema.Literal("function"),
  function: Schema.Struct({ name: Schema.String, arguments: Schema.String }),
})

const ChatSystemRequestMessageSchema = Schema.Struct({
  role: Schema.Literal("system"),
  content: Schema.String,
})
const ChatUserRequestMessageSchema = Schema.Struct({
  role: Schema.Literal("user"),
  content: Schema.Union(Schema.String, Schema.Array(ChatContentPartSchema)),
})
const ChatAssistantRequestMessageSchema = Schema.Struct({
  role: Schema.Literal("assistant"),
  content: Schema.String,
  reasoning_content: Schema.optionalWith(
    Schema.String,
    { exact: true, as: "Option" },
  ),
  tool_calls: Schema.optionalWith(
    Schema.Array(ChatToolCallSchema),
    { exact: true, as: "Option" },
  ),
})
const ChatToolRequestMessageSchema = Schema.Struct({
  role: Schema.Literal("tool"),
  tool_call_id: Schema.String,
  content: Schema.Union(Schema.String, Schema.Array(ChatContentPartSchema)),
})
const ChatCompletionsRequestMessageSchema = Schema.Union(
  ChatSystemRequestMessageSchema,
  ChatUserRequestMessageSchema,
  ChatAssistantRequestMessageSchema,
  ChatToolRequestMessageSchema,
)

const ChatNamedFunctionToolChoiceSchema = Schema.Struct({
  type: Schema.Literal("function"),
  function: Schema.Struct({ name: Schema.String }),
})
const ChatAllowedToolsToolChoiceSchema = Schema.Struct({
  type: Schema.Literal("allowed_tools"),
  allowed_tools: Schema.Struct({
    mode: Schema.Literal("auto", "required"),
    tools: Schema.Array(Schema.Struct({
      type: Schema.Literal("function"),
      function: Schema.Struct({ name: Schema.String }),
    })),
  }),
})
export const ChatToolChoiceSchema = Schema.Union(
  Schema.Literal("auto", "none", "required"),
  ChatNamedFunctionToolChoiceSchema,
  ChatAllowedToolsToolChoiceSchema,
  Schema.Struct({ type: Schema.Literal("grammar"), grammar: Schema.String }),
)
const JsonSchemaObjectSchema = Schema.declare<JsonSchemaObject>(
  (value): value is JsonSchemaObject => Schema.is(JsonRecordSchema)(value),
)
export const ChatToolSchema = Schema.Struct({
  type: Schema.Literal("function"),
  function: Schema.Struct({
    name: Schema.String,
    description: Schema.String,
    parameters: JsonSchemaObjectSchema,
  }),
})

export const ChatCompletionsPromptSchema = Schema.Struct({
  model: Schema.String,
  messages: Schema.Array(ChatCompletionsRequestMessageSchema),
  tools: Schema.optionalWith(
    Schema.Array(ChatToolSchema),
    { exact: true, as: "Option" },
  ),
})

const ChatCompletionsRequestValueFields = {
  model: Schema.String,
  messages: Schema.Array(ChatCompletionsRequestMessageSchema),
  tools: Schema.optionalWith(
    Schema.Array(ChatToolSchema),
    { exact: true, as: "Option" },
  ),
  tool_choice: Schema.optionalWith(ChatToolChoiceSchema, { exact: true, as: "Option" }),
  max_tokens: Schema.optionalWith(Schema.Number, { exact: true, as: "Option" }),
  stop: Schema.optionalWith(Schema.Array(Schema.String), { exact: true, as: "Option" }),
  temperature: Schema.optionalWith(Schema.Number, { exact: true, as: "Option" }),
  top_p: Schema.optionalWith(Schema.Number, { exact: true, as: "Option" }),
  reasoning_effort: Schema.optionalWith(
    Schema.String,
    { exact: true, as: "Option" },
  ),
  logprobs: Schema.optionalWith(Schema.Boolean, { exact: true, as: "Option" }),
  top_logprobs: Schema.optionalWith(Schema.Number, { exact: true, as: "Option" }),
  stream: Schema.Literal(true),
  stream_options: Schema.optionalWith(
    Schema.Struct({ include_usage: Schema.Boolean }),
    { exact: true, as: "Option" },
  ),
} as const

const standardRequestProperties = new Set(Object.keys(ChatCompletionsRequestValueFields))
export const ChatCompletionsRequestExtensionsSchema = JsonRecordSchema.pipe(Schema.filter(
  (extensions) => Object.keys(extensions).every(
    (property) => !standardRequestProperties.has(property),
  ),
  { message: () => "request extensions cannot replace standard Chat Completions properties" },
))

const ChatCompletionsRequestValueSchema = Schema.Struct({
  ...ChatCompletionsRequestValueFields,
  extensions: ChatCompletionsRequestExtensionsSchema,
})

const ChatCompletionsRequestWireCoreSchema = Schema.encodedSchema(
  Schema.Struct(ChatCompletionsRequestValueFields),
)

const ChatCompletionsRequestWireSchema = Schema.extend(
  ChatCompletionsRequestWireCoreSchema,
  JsonRecordSchema,
)

export const ChatCompletionsRequestSchema = Schema.transform(
  ChatCompletionsRequestWireSchema,
  ChatCompletionsRequestValueSchema,
  {
    strict: true,
    decode: (wire) => {
      const extensions: Record<string, JsonValue> = {}
      for (const [property, value] of Object.entries(wire)) {
        if (!standardRequestProperties.has(property)) extensions[property] = value
      }
      return {
        ...wire,
        extensions,
      }
    },
    encode: (value) => {
      const { extensions, ...standard } = value
      return { ...extensions, ...standard }
    },
  },
)

export type ChatTextContentPart = Schema.Schema.Encoded<typeof ChatTextContentPartSchema>
export type ChatImageUrlContentPart = Schema.Schema.Encoded<typeof ChatImageUrlContentPartSchema>
export type ChatContentPart = Schema.Schema.Encoded<typeof ChatContentPartSchema>
export type ChatToolCall = Schema.Schema.Encoded<typeof ChatToolCallSchema>
export type ChatCompletionsRequestMessage = Schema.Schema.Encoded<typeof ChatCompletionsRequestMessageSchema>
export type ChatMessage = ChatCompletionsRequestMessage
export type ChatMessageRole = ChatCompletionsRequestMessage["role"]
export type ChatNamedFunctionToolChoice = Schema.Schema.Encoded<typeof ChatNamedFunctionToolChoiceSchema>
export type ChatAllowedToolsToolChoice = Schema.Schema.Encoded<typeof ChatAllowedToolsToolChoiceSchema>
export type ChatToolChoice = Schema.Schema.Encoded<typeof ChatToolChoiceSchema>
export type ChatTool = Schema.Schema.Encoded<typeof ChatToolSchema>
export type ChatCompletionsPrompt = Schema.Schema.Encoded<typeof ChatCompletionsPromptSchema>
export type ChatCompletionsRequest = Schema.Schema.Encoded<typeof ChatCompletionsRequestSchema>

export const decodeChatCompletionsRequest = Schema.decodeUnknown(ChatCompletionsRequestSchema)

export const encodeChatCompletionsRequest = Schema.encode(ChatCompletionsRequestSchema)

export const finalizeChatCompletionsRequest = (
  draft: unknown,
): Effect.Effect<ChatCompletionsRequest, ParseResult.ParseError> =>
  decodeChatCompletionsRequest(draft).pipe(Effect.flatMap(encodeChatCompletionsRequest))
