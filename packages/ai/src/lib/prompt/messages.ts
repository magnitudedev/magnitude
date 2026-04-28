import { Schema } from "effect"
import type { ToolCallId } from "./ids"
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
  readonly reasoning?: string
  readonly text?: string
  readonly toolCalls?: readonly ToolCallPart[]
}

export interface ToolResultMessage {
  readonly _tag: "ToolResultMessage"
  readonly toolCallId: ToolCallId
  readonly toolName: string
  readonly parts: readonly ToolResultPart[]
}

const UserPartSchema = Schema.Union(TextPartSchema, ImagePartSchema)

export const UserMessageSchema = Schema.TaggedStruct("UserMessage", {
  parts: Schema.Array(UserPartSchema),
})

export const AssistantMessageSchema = Schema.TaggedStruct("AssistantMessage", {
  reasoning: Schema.optional(Schema.String),
  text: Schema.optional(Schema.String),
  toolCalls: Schema.optional(Schema.Array(ToolCallPartSchema)),
})

export const ToolResultMessageSchema = Schema.TaggedStruct("ToolResultMessage", {
  toolCallId: Schema.String,
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
