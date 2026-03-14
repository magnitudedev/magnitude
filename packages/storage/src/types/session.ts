import { Schema } from 'effect'

export const StoredSessionMetaSchema = Schema.Struct({
  sessionId: Schema.String,
  created: Schema.String,
  updated: Schema.String,
  chatName: Schema.String,
  workingDirectory: Schema.String,
  gitBranch: Schema.optionalWith(Schema.NullishOr(Schema.String), { default: () => null }),
  firstUserMessage: Schema.optionalWith(Schema.NullishOr(Schema.String), { default: () => null }),
  lastMessage: Schema.optionalWith(Schema.NullishOr(Schema.String), { default: () => null }),
  messageCount: Schema.Number,
})
export interface StoredSessionMeta extends Omit<Schema.Schema.Type<typeof StoredSessionMetaSchema>, 'gitBranch' | 'firstUserMessage' | 'lastMessage'> {
  gitBranch: string | null
  firstUserMessage: string | null
  lastMessage: string | null
}

export const MemoryExtractionJobRecordSchema = Schema.Struct({
  jobId: Schema.String,
  sessionId: Schema.String,
  cwd: Schema.String,
  eventsPath: Schema.String,
  memoryPath: Schema.String,
  createdAt: Schema.String,
  attempts: Schema.Number,
  status: Schema.Union(Schema.Literal('pending'), Schema.Literal('running')),
})
export type MemoryExtractionJobRecord = Schema.Schema.Type<typeof MemoryExtractionJobRecordSchema>

export interface SessionDiscoveryOptions {
  readonly timestampOnly?: boolean
}