import { Schema } from 'effect'

export const EditDiffSchema = Schema.Struct({
  startLine: Schema.Number,
  removedLines: Schema.Array(Schema.String),
  addedLines: Schema.Array(Schema.String),
  contextBefore: Schema.Array(Schema.String),
  contextAfter: Schema.Array(Schema.String),
})
export type EditDiff = typeof EditDiffSchema.Type
