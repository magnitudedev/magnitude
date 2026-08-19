import { Effect, Option, Schema } from 'effect'

import { Version } from '../services/version'
import { ProjectIdSchema } from '@magnitudedev/acn-protocol'

const LegacySerializedPinnedAtSchema = Schema.Union(
  Schema.String,
  Schema.TaggedStruct('None', {
    _id: Schema.Literal('Option'),
  }),
  Schema.TaggedStruct('Some', {
    _id: Schema.Literal('Option'),
    value: Schema.String,
  }),
)

const RawStoredSessionMetaSchema = Schema.Struct({
  sessionId: Schema.String,
  projectId: ProjectIdSchema,
  created: Schema.String,
  updated: Schema.String,
  chatName: Schema.String,
  workingDirectory: Schema.String,
  archived: Schema.optionalWith(Schema.Boolean, { as: 'Option', exact: true }),
  /** Input-only migration provenance. Newly encoded metadata omits this field. */
  sidebarOpen: Schema.optionalWith(Schema.Boolean, { as: 'Option', exact: true }),
  /** Accepts malformed Option JSON written before session metadata used schema-aware encoding. */
  pinnedAt: Schema.optionalWith(LegacySerializedPinnedAtSchema, { as: 'Option', exact: true }),
  visibility: Schema.optionalWith(Schema.Union(Schema.Literal('draft'), Schema.Literal('visible')), { default: () => 'visible' }),
  initialVersion: Schema.optional(Schema.String),
  lastActiveVersion: Schema.optional(Schema.String),
  gitBranch: Schema.optionalWith(Schema.NullishOr(Schema.String), { default: () => null }),
  firstUserMessage: Schema.optionalWith(Schema.NullishOr(Schema.String), { default: () => null }),
  lastMessage: Schema.optionalWith(Schema.NullishOr(Schema.String), { default: () => null }),
  messageCount: Schema.optionalWith(Schema.Number, { default: () => 0 }),
})

const DecodedStoredSessionMetaSchema = Schema.Struct({
  sessionId: Schema.String,
  projectId: ProjectIdSchema,
  created: Schema.String,
  updated: Schema.String,
  chatName: Schema.String,
  workingDirectory: Schema.String,
  archived: Schema.Boolean,
  pinnedAt: Schema.OptionFromSelf(Schema.String),
  visibility: Schema.Union(Schema.Literal('draft'), Schema.Literal('visible')),
  initialVersion: Schema.String,
  lastActiveVersion: Schema.String,
  gitBranch: Schema.NullOr(Schema.String),
  firstUserMessage: Schema.NullOr(Schema.String),
  lastMessage: Schema.NullOr(Schema.String),
  messageCount: Schema.Number,
})

type RawStoredSessionMeta = typeof RawStoredSessionMetaSchema.Type
type DecodedStoredSessionMeta = typeof DecodedStoredSessionMetaSchema.Type

const decodeStoredSessionMeta = (
  raw: RawStoredSessionMeta,
  version: string,
): DecodedStoredSessionMeta => {
  const { archived, sidebarOpen, ...rest } = raw
  const isArchived = Option.match(archived, {
    onNone: () => Option.match(sidebarOpen, {
      onNone: () => false,
      onSome: (open) => !open,
    }),
    onSome: (value) => value,
  })
  return {
    ...rest,
    archived: isArchived,
    pinnedAt: isArchived
      ? Option.none()
      : Option.flatMap(raw.pinnedAt, (value) =>
          typeof value === 'string'
            ? Option.some(value)
            : value._tag === 'Some'
            ? Option.some(value.value)
            : Option.none()),
    initialVersion: raw.initialVersion ?? version,
    lastActiveVersion: raw.lastActiveVersion ?? version,
    gitBranch: raw.gitBranch ?? null,
    firstUserMessage: raw.firstUserMessage ?? null,
    lastMessage: raw.lastMessage ?? null,
  }
}

const encodeStoredSessionMeta = (
  meta: DecodedStoredSessionMeta,
): RawStoredSessionMeta => {
  const { archived, ...rest } = meta
  return {
    ...rest,
    archived: Option.some(archived),
    sidebarOpen: Option.none(),
    pinnedAt: archived ? Option.none() : meta.pinnedAt,
  }
}

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
      strict: true,
      decode: (raw) => Effect.succeed(decodeStoredSessionMeta(raw, version)),
      encode: (_encoded, _options, _ast, meta) =>
        Effect.succeed(encodeStoredSessionMeta(meta)),
    }
  )
}

/** Legacy export for backward compatibility — requires Version in context. */
export const StoredSessionMetaSchema = Schema.transformOrFail(
  RawStoredSessionMetaSchema,
  DecodedStoredSessionMetaSchema,
  {
    strict: true,
    decode: (raw) =>
      Effect.map(Version, (version) =>
        decodeStoredSessionMeta(raw, version.getVersion())),
    encode: (_encoded, _options, _ast, meta) =>
      Effect.succeed(encodeStoredSessionMeta(meta)),
  }
)
export type StoredSessionMeta = Schema.Schema.Type<typeof StoredSessionMetaSchema>

export const LegacyStoredSessionProjectRecordSchema = Schema.Struct({
  sessionId: Schema.String,
  workingDirectory: Schema.String,
  projectId: Schema.optionalWith(ProjectIdSchema, { as: 'Option', exact: true }),
})
export type LegacyStoredSessionProjectRecord =
  Schema.Schema.Type<typeof LegacyStoredSessionProjectRecordSchema>

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
  readonly cwd: string
  readonly sessionIds: readonly string[]
}

export interface SessionDiscoveryOptions {
  readonly timestampOnly?: boolean
}
