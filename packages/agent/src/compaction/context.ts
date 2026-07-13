import { Context, Ref, Schema } from 'effect'

export const CompactResultSchema = Schema.Struct({
  summary: Schema.String,
  reflection: Schema.String,
  files: Schema.Array(Schema.Struct({
    path: Schema.String,
    content: Schema.String,
  })),
})

export type CompactResult = typeof CompactResultSchema.Type

export interface CompactionContext {
  readonly isCompacting: true
  readonly resultRef: Ref.Ref<CompactResult | null>
  readonly maxPayloadTokens: number
}

export class CompactionContextTag extends Context.Tag('CompactionContext')<
  CompactionContextTag,
  CompactionContext
>() {}
