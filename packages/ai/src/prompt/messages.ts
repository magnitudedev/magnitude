import { Schema, Option } from "effect"
import type { ProviderToolCallId, ToolCallId } from "./ids"
import {
  ImagePartSchema,
  JsonValueSchema,
  TextPartSchema,
  ToolCallPartSchema,
  type ImagePart,
  type JsonValue,
  type TextPart,
  type ToolCallPart,
} from "./parts"

export type UserPart = TextPart | ImagePart
export type ToolResultPart = TextPart | ImagePart

export interface UserMessage {
  readonly _tag: "UserMessage"
  readonly parts: readonly UserPart[]
}

export interface AssistantMessage {
  readonly _tag: "AssistantMessage"
  readonly reasoning: Option.Option<string>
  /** Provider-native reasoning blocks required for lossless tool-call continuation. */
  readonly reasoningDetails: readonly JsonValue[]
  readonly text: Option.Option<string>
  readonly toolCalls: Option.Option<readonly ToolCallPart[]>
}

export interface ToolResultMessage {
  readonly _tag: "ToolResultMessage"
  readonly toolCallId: ToolCallId
  readonly providerToolCallId: ProviderToolCallId
  readonly toolName: string
  readonly parts: readonly ToolResultPart[]
}

export const UserPartSchema = Schema.Union(TextPartSchema, ImagePartSchema)

export const UserMessageSchema = Schema.TaggedStruct("UserMessage", {
  parts: Schema.Array(UserPartSchema),
})

export const AssistantMessageSchema = Schema.TaggedStruct("AssistantMessage", {
  reasoning: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  reasoningDetails: Schema.optionalWith(Schema.Array(JsonValueSchema), { default: () => [] }),
  text: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  toolCalls: Schema.optionalWith(Schema.Array(ToolCallPartSchema), { as: "Option", exact: true }),
})

const REASONING_DETAIL_FRAGMENT_FIELDS = ["text", "summary", "data"] as const

function isJsonRecord(value: JsonValue): value is { readonly [key: string]: JsonValue } {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

/** Coalesce provider streaming deltas into the exact reasoning block sequence to replay. */
export function mergeReasoningDetails(
  current: readonly JsonValue[],
  deltas: readonly JsonValue[],
): readonly JsonValue[] {
  const result = [...current]
  for (const delta of deltas) {
    if (!isJsonRecord(delta) || typeof delta.index !== "number") {
      result.push(delta)
      continue
    }
    const existingIndex = result.findIndex((candidate) =>
      isJsonRecord(candidate)
      && candidate.index === delta.index
      && candidate.type === delta.type
    )
    if (existingIndex < 0) {
      result.push(delta)
      continue
    }

    const previous = result[existingIndex]
    if (!isJsonRecord(previous)) continue
    const merged: { [key: string]: JsonValue } = { ...previous, ...delta }
    for (const field of REASONING_DETAIL_FRAGMENT_FIELDS) {
      const priorFragment = previous[field]
      const nextFragment = delta[field]
      if (typeof priorFragment === "string" && typeof nextFragment === "string") {
        merged[field] = priorFragment + nextFragment
      }
    }
    result[existingIndex] = merged
  }
  return result
}

export const ToolResultMessageSchema = Schema.TaggedStruct("ToolResultMessage", {
  toolCallId: Schema.String,
  providerToolCallId: Schema.String,
  toolName: Schema.String,
  parts: Schema.Array(UserPartSchema),
})

export const MessageSchema = Schema.Union(
  UserMessageSchema,
  AssistantMessageSchema,
  ToolResultMessageSchema,
)

export type Message = UserMessage | AssistantMessage | ToolResultMessage
export type TerminalMessage = UserMessage | ToolResultMessage
