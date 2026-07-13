import { Effect, Schema } from 'effect'

import { Version } from '../services/version'

const RawStoredSessionMetaSchema = Schema.Struct({
  sessionId: Schema.String,
  created: Schema.String,
  updated: Schema.String,
  chatName: Schema.String,
  workingDirectory: Schema.String,
  visibility: Schema.optionalWith(Schema.Union(Schema.Literal('draft'), Schema.Literal('visible')), { default: () => 'visible' }),
  initialVersion: Schema.optional(Schema.String),
  lastActiveVersion: Schema.optional(Schema.String),
  gitBranch: Schema.optionalWith(Schema.NullishOr(Schema.String), { default: () => null }),
  firstUserMessage: Schema.optionalWith(Schema.NullishOr(Schema.String), { default: () => null }),
  lastMessage: Schema.optionalWith(Schema.NullishOr(Schema.String), { default: () => null }),
  messageCount: Schema.Number,
})

const DecodedStoredSessionMetaSchema = Schema.Struct({
  sessionId: Schema.String,
  created: Schema.String,
  updated: Schema.String,
  chatName: Schema.String,
  workingDirectory: Schema.String,
  visibility: Schema.Union(Schema.Literal('draft'), Schema.Literal('visible')),
  initialVersion: Schema.String,
  lastActiveVersion: Schema.String,
  gitBranch: Schema.NullOr(Schema.String),
  firstUserMessage: Schema.NullOr(Schema.String),
  lastMessage: Schema.NullOr(Schema.String),
  messageCount: Schema.Number,
})

/**
 * Creates a StoredSessionMetaSchema that fills in default version values
 * from the provided version string, instead of requiring the Version service
 * from the Effect context.
 */
export function makeStoredSessionMetaSchema(version: string) {
  return Schema.transformOrFail(
    RawStoredSessionMetaSchema,
    DecodedStoredSessionMetaSchema,
    {
      decode: (raw) =>
        Effect.succeed({
          ...raw,
          initialVersion: raw.initialVersion ?? version,
          lastActiveVersion: raw.lastActiveVersion ?? version,
          gitBranch: raw.gitBranch ?? null,
          firstUserMessage: raw.firstUserMessage ?? null,
          lastMessage: raw.lastMessage ?? null,
        }),
      encode: (meta) => Effect.succeed({ ...meta }),
    }
  )
}

/** Legacy export for backward compatibility — requires Version in context. */
export const StoredSessionMetaSchema = Schema.transformOrFail(
  RawStoredSessionMetaSchema,
  DecodedStoredSessionMetaSchema,
  {
    decode: (raw) =>
      Effect.map(Version, (version) => ({
        ...raw,
        initialVersion: raw.initialVersion ?? version.getVersion(),
        lastActiveVersion: raw.lastActiveVersion ?? version.getVersion(),
        gitBranch: raw.gitBranch ?? null,
        firstUserMessage: raw.firstUserMessage ?? null,
        lastMessage: raw.lastMessage ?? null,
      })),
    encode: (meta) => Effect.succeed({ ...meta }),
  }
)
export type StoredSessionMeta = Schema.Schema.Type<typeof StoredSessionMetaSchema>

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

export interface CwdIndex {
  cwd: string
  sessionIds: string[]
}

export interface SessionDiscoveryOptions {
  readonly timestampOnly?: boolean
}
