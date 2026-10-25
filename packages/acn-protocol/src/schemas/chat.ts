import { Option, Schema } from "effect"
import { ModelLoadDetails, OpenAiModelsResponse } from "@magnitudedev/icn-protocol/schemas"

const id = <B extends string>(brand: B) =>
  Schema.NonEmptyString.pipe(Schema.maxLength(128), Schema.brand(brand))
export const ChatConversationId = id("ChatConversationId")
export type ChatConversationId = typeof ChatConversationId.Type
export const ChatMessageId = id("ChatMessageId")
export type ChatMessageId = typeof ChatMessageId.Type
export const ChatGenerationId = id("ChatGenerationId")
export type ChatGenerationId = typeof ChatGenerationId.Type
export const ChatCommandId = id("ChatCommandId")
export type ChatCommandId = typeof ChatCommandId.Type
export const ChatModelId = Schema.NonEmptyString.pipe(
  Schema.maxLength(1024),
  Schema.brand("ChatModelId")
)
export type ChatModelId = typeof ChatModelId.Type
const finite = Schema.Number.pipe(Schema.finite(), Schema.nonNegative())
const count = finite.pipe(Schema.int())
const optional = <A, I>(schema: Schema.Schema<A, I>) =>
  Schema.optionalWith(schema, { as: "Option", exact: true })
const range = (min: number, max: number) =>
  Schema.Number.pipe(Schema.finite(), Schema.between(min, max))
export const ChatSettings = Schema.Struct({
  modelId: optional(ChatModelId),
  systemPrompt: Schema.String.pipe(Schema.maxLength(65_536)),
  reasoningEffort: optional(Schema.String.pipe(Schema.maxLength(32))),
  temperature: optional(range(0, 2)),
  maxOutputTokens: optional(
    Schema.Number.pipe(Schema.int(), Schema.between(1, 2_000_000))
  ),
  topP: optional(
    Schema.Number.pipe(
      Schema.finite(),
      Schema.greaterThan(0),
      Schema.lessThanOrEqualTo(1)
    )
  ),
  topK: optional(Schema.Number.pipe(Schema.int(), Schema.between(0, 1 << 24))),
  minP: optional(range(0, 1)),
  repetitionPenalty: optional(
    Schema.Number.pipe(
      Schema.finite(),
      Schema.greaterThan(0),
      Schema.lessThanOrEqualTo(100)
    )
  ),
  presencePenalty: optional(range(-2, 2)),
  frequencyPenalty: optional(range(-2, 2)),
  seed: optional(
    Schema.Number.pipe(Schema.int(), Schema.between(0, 0xffffffff))
  ),
  stop: Schema.Array(
    Schema.String.pipe(Schema.minLength(1), Schema.maxLength(1024))
  ).pipe(Schema.maxItems(16)),
})
export type ChatSettings = typeof ChatSettings.Type
export const defaultChatSettings = Schema.decodeUnknownSync(ChatSettings)({
  systemPrompt: "You are a helpful assistant.",
  stop: [],
})
export const ChatMetrics = Schema.Struct({
  inputTokens: optional(count),
  cachedTokens: optional(count),
  outputTokens: optional(count),
  decodeMs: optional(finite),
  tokensPerSecond: optional(finite),
  promptMs: optional(finite),
  promptTokensPerSecond: optional(finite),
  ttftMs: optional(finite),
  engineTtftMs: optional(finite),
  firstAnswerMs: optional(finite),
  reasoningMs: optional(finite),
  totalMs: optional(finite),
})
export type ChatMetrics = typeof ChatMetrics.Type
export const emptyChatMetrics = Schema.decodeUnknownSync(ChatMetrics)({})
export const ChatPhase = Schema.Literal(
  "starting",
  "model_queued",
  "preparing_model",
  "generating",
  "queued",
  "optimizing",
  "loading",
  "finalizing",
  "preparing",
  "prefill",
  "thinking",
  "responding"
)
export type ChatPhase = typeof ChatPhase.Type
export const ChatActivity = Schema.Struct({
  phase: ChatPhase,
  since: finite,
  fraction: optional(range(0, 1)),
  completed: optional(count),
  total: optional(count),
  cached: optional(count),
  loadDetails: optional(ModelLoadDetails),
})
export type ChatActivity = typeof ChatActivity.Type
export const ChatGenerationStatus = Schema.Literal(
  "running",
  "completed",
  "length",
  "stopped",
  "interrupted",
  "failed"
)
export const ChatGeneration = Schema.Struct({
  id: ChatGenerationId,
  commandId: ChatCommandId,
  messageId: ChatMessageId,
  modelId: ChatModelId,
  settings: ChatSettings,
  status: ChatGenerationStatus,
  createdAt: finite,
  updatedAt: finite,
  content: Schema.String,
  reasoning: Schema.String,
  activity: optional(ChatActivity),
  metrics: ChatMetrics,
  error: optional(Schema.String),
})
export type ChatGeneration = typeof ChatGeneration.Type
export const ChatMessage = Schema.Struct({
  id: ChatMessageId,
  role: Schema.Literal("user", "assistant"),
  content: Schema.String,
  generation: optional(ChatGeneration),
})
export type ChatMessage = typeof ChatMessage.Type
export const ChatConversation = Schema.Struct({
  id: ChatConversationId,
  title: Schema.String,
  createdAt: finite,
  updatedAt: finite,
  settings: ChatSettings,
  messages: Schema.Array(ChatMessage),
})
export type ChatConversation = typeof ChatConversation.Type
export const ChatSummary = ChatConversation.omit("messages").pipe(
  Schema.extend(Schema.Struct({ messageCount: count, active: Schema.Boolean }))
)
export type ChatSummary = typeof ChatSummary.Type
export const ChatHistory = Schema.Struct({
  conversations: Schema.Array(ChatSummary),
  hasMore: Schema.Boolean,
})
export const ChatUpdate = Schema.Union(
  Schema.TaggedStruct("Snapshot", { conversation: ChatConversation }),
  Schema.TaggedStruct("Generation", { generation: ChatGeneration }),
  Schema.TaggedStruct("Deleted", { conversationId: ChatConversationId })
)
export type ChatUpdate = typeof ChatUpdate.Type
export class ChatFailed extends Schema.TaggedError<ChatFailed>()("ChatFailed", {
  code: Schema.Literal(
    "NotFound",
    "Busy",
    "InvalidRequest",
    "Storage",
    "Inference"
  ),
  message: Schema.String,
}) {}
export const ChatModels = OpenAiModelsResponse

/** Cumulative output replaces only its message; prior transcript objects stay stable. */
export const applyChatGeneration = (
  conversation: ChatConversation,
  generation: ChatGeneration
): ChatConversation => ({
  ...conversation,
  updatedAt: Math.max(conversation.updatedAt, generation.updatedAt),
  messages: conversation.messages.map((message) =>
    message.id === generation.messageId
      ? {
          ...message,
          content: generation.content,
          generation: Option.some(generation),
        }
      : message
  ),
})

export const applyChatUpdate = (
  conversation: ChatConversation | null,
  update: ChatUpdate
): ChatConversation | null =>
  update._tag === "Snapshot"
    ? update.conversation
    : update._tag === "Deleted"
    ? null
    : conversation
    ? applyChatGeneration(conversation, update.generation)
    : null
