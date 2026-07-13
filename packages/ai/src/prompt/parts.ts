import { Schema } from "effect"
import { JsonValueSchema, type JsonValue } from "@magnitudedev/utils/schema"
export {
  JsonRecordSchema,
  JsonValueSchema,
  type JsonPrimitive,
  type JsonRecord,
  type JsonValue,
} from "@magnitudedev/utils/schema"
import { ProviderToolCallIdSchema, ToolCallIdSchema, type ProviderToolCallId, type ToolCallId } from "./ids"

export interface TextPart {
  readonly _tag: "TextPart"
  readonly text: string
}

export type ImageMediaType = 'image/png' | 'image/jpeg' | 'image/webp' | 'image/gif'

export interface ImagePart {
  readonly _tag: "ImagePart"
  readonly data: string
  readonly mediaType: ImageMediaType
  readonly dimensions?: { readonly width: number; readonly height: number }
}

export interface ToolCallPart {
  readonly _tag: "ToolCallPart"
  readonly id: ToolCallId
  readonly providerToolCallId: ProviderToolCallId
  readonly name: string
  readonly input: JsonValue
}

export const TextPartSchema = Schema.TaggedStruct("TextPart", {
  text: Schema.String,
})

export const ImagePartSchema = Schema.TaggedStruct("ImagePart", {
  data: Schema.String,
  mediaType: Schema.Literal('image/png', 'image/jpeg', 'image/webp', 'image/gif'),
})

export const ToolCallPartSchema = Schema.TaggedStruct("ToolCallPart", {
  id: ToolCallIdSchema,
  providerToolCallId: ProviderToolCallIdSchema,
  name: Schema.String,
  input: JsonValueSchema,
})

export type PromptPart = TextPart | ImagePart | ToolCallPart
