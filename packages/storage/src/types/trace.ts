import { Schema } from 'effect'

export const StoredTraceSessionMetaSchema = Schema.Struct({
  sessionId: Schema.String,
  created: Schema.String,
  cwd: Schema.optionalWith(Schema.NullishOr(Schema.String), { default: () => null }),
  platform: Schema.optionalWith(Schema.NullishOr(Schema.String), { default: () => null }),
  gitBranch: Schema.optionalWith(Schema.NullishOr(Schema.String), { default: () => null }),
  chatName: Schema.optionalWith(Schema.NullishOr(Schema.String), { default: () => null }),
})
export interface StoredTraceSessionMeta extends Schema.Schema.Type<typeof StoredTraceSessionMetaSchema> {
  readonly cwd: string | null
  readonly platform: string | null
  readonly gitBranch: string | null
  readonly chatName: string | null
}
