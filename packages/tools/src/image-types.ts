import { Schema } from '@effect/schema'

export type ImageMediaType = 'image/png' | 'image/jpeg' | 'image/webp' | 'image/gif'

export interface ToolImageValue {
  readonly base64: string
  readonly mediaType: ImageMediaType
  readonly width: number
  readonly height: number
}

export type ContentPart =
  | { readonly type: 'text'; readonly text: string }
  | { readonly type: 'image'; readonly base64: string; readonly mediaType: ImageMediaType; readonly width: number; readonly height: number }

export const ToolImageSchema = Schema.Struct({
  base64: Schema.String,
  mediaType: Schema.Literal('image/png', 'image/jpeg', 'image/webp', 'image/gif'),
  width: Schema.Number,
  height: Schema.Number,
}).annotations({ identifier: 'ToolImage' })