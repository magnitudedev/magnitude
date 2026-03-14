import { Context, Effect } from 'effect'

import type { MemoryExtractionJobRecord, StoredSessionMeta } from '../types'

export interface SessionStorageShape {
  readonly paths: {
    readonly root: string
    readonly sessionsRoot: string
    readonly pendingMemoryExtractionRoot: string
    readonly sessionDir: (sessionId: string) => string
    readonly sessionMetaFile: (sessionId: string) => string
    readonly sessionEventsFile: (sessionId: string) => string
    readonly sessionLogFile: (sessionId: string) => string
    readonly sessionArtifactsDir: (sessionId: string) => string
    readonly sessionArtifactFile: (sessionId: string, artifactName: string) => string
    readonly pendingMemoryJobFile: (jobId: string) => string
  }

  readonly createTimestampSessionId: (now?: Date) => string

  readonly listSessionIds: (options?: {
    readonly timestampOnly?: boolean
  }) => Effect.Effect<string[]>

  readonly findLatestSessionId: (options?: {
    readonly timestampOnly?: boolean
  }) => Effect.Effect<string | null>

  readonly readMeta: (
    sessionId: string
  ) => Effect.Effect<StoredSessionMeta | null>

  readonly writeMeta: (
    sessionId: string,
    meta: StoredSessionMeta
  ) => Effect.Effect<void>

  readonly updateMeta: (
    sessionId: string,
    updater: (current: StoredSessionMeta | null) => StoredSessionMeta
  ) => Effect.Effect<StoredSessionMeta>

  readonly readEvents: <T>(
    sessionId: string
  ) => Effect.Effect<T[]>

  readonly appendEvents: <T>(
    sessionId: string,
    events: readonly T[]
  ) => Effect.Effect<void>

  readonly readEventsFromPath: <T>(
    eventsPath: string
  ) => Effect.Effect<T[]>

  readonly appendLogs: <T>(
    sessionId: string,
    entries: readonly T[]
  ) => Effect.Effect<void>

  readonly clearLog: (
    sessionId: string
  ) => Effect.Effect<void>

  readonly writeArtifact: (
    sessionId: string,
    name: string,
    content: string,
    options?: { readonly extension?: string }
  ) => Effect.Effect<string>

  readonly createMemoryExtractionJobRecord: (params: {
    readonly sessionId: string
    readonly cwd: string
    readonly eventsPath: string
    readonly memoryPath: string
    readonly now?: Date
    readonly createId?: () => string
  }) => MemoryExtractionJobRecord

  readonly writePendingMemoryJob: (
    job: MemoryExtractionJobRecord
  ) => Effect.Effect<string>

  readonly listPendingMemoryJobFiles: () => Effect.Effect<string[]>

  readonly listPendingMemoryJobIds: () => Effect.Effect<string[]>

  readonly readPendingMemoryJob: (
    input: { readonly jobId: string } | { readonly filePath: string }
  ) => Effect.Effect<MemoryExtractionJobRecord>

  readonly markPendingMemoryJobRunning: (
    input: { readonly jobId: string } | { readonly filePath: string },
    job?: MemoryExtractionJobRecord
  ) => Effect.Effect<MemoryExtractionJobRecord>

  readonly markPendingMemoryJobPending: (
    input: { readonly jobId: string } | { readonly filePath: string },
    job?: MemoryExtractionJobRecord
  ) => Effect.Effect<MemoryExtractionJobRecord>

  readonly removePendingMemoryJob: (
    input: { readonly jobId: string } | { readonly filePath: string }
  ) => Effect.Effect<void>

  readonly resolvePendingMemoryJobPath: (
    jobId: string
  ) => string
}

export class SessionStorage extends Context.Tag('SessionStorage')<
  SessionStorage,
  SessionStorageShape
>() {}