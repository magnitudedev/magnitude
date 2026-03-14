import { Schema } from 'effect'

export const StoredTraceSessionMetaSchema = Schema.Struct({
  sessionId: Schema.String,
  created: Schema.String,
  cwd: Schema.optionalWith(Schema.NullishOr(Schema.String), { default: () => null }),
  platform: Schema.optionalWith(Schema.NullishOr(Schema.String), { default: () => null }),
  gitBranch: Schema.optionalWith(Schema.NullishOr(Schema.String), { default: () => null }),
})
export interface StoredTraceSessionMeta
  extends Omit<Schema.Schema.Type<typeof StoredTraceSessionMetaSchema>, 'cwd' | 'platform' | 'gitBranch'>,
    Record<string, unknown> {
  cwd: string | null
  platform: string | null
  gitBranch: string | null
}