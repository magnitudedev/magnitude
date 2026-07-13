import { Schema, Option } from "effect"
import type { ProviderToolCallId, ToolCallId } from "./ids"
import {
  ImagePartSchema,
  TextPartSchema,
  ToolCallPartSchema,
  type ImagePart,
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
  text: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  toolCalls: Schema.optionalWith(Schema.Array(ToolCallPartSchema), { as: "Option", exact: true }),
})

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
