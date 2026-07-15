import type { PlatformError } from '@effect/platform/Error'
import { Context, Effect, Schema } from 'effect'

import type { JsonError, JsonLinesError } from '../io/storage'
import type { CwdIndex, MemoryExtractionJobRecord, StoredSessionMeta } from '../types'

export interface StoredEventCursor {
  readonly index: number
  readonly timestamp: number
}

export interface StoredAddressedEntry {
  readonly value: unknown
}

export interface StoredAddressedEntryStats {
  readonly storedBytes: number
}

export interface SessionStorageShape {
  readonly paths: {
    readonly root: string
    readonly sessionsRoot: string
    readonly pendingMemoryExtractionRoot: string
    readonly sessionDir: (sessionId: string) => string
    readonly sessionMetaFile: (sessionId: string) => string
    readonly sessionEventsFile: (sessionId: string) => string
    readonly sessionProjectionSnapshotFile: (sessionId: string) => string
    readonly sessionAddressedEntriesRoot: (sessionId: string) => string
    readonly sessionAddressedEntryFile: (
      sessionId: string,
      namespace: string,
      address: string
    ) => string
    readonly sessionLogFile: (sessionId: string) => string
    readonly sessionScratchpad: (sessionId: string) => string
    readonly pendingMemoryJobFile: (jobId: string) => string
  }

  readonly createTimestampSessionId: () => string

  readonly listSessionIds: (options?: {
    readonly timestampOnly?: boolean
  }) => Effect.Effect<string[], PlatformError | JsonError>

  readonly findLatestSessionId: (options?: {
    readonly timestampOnly?: boolean
  }) => Effect.Effect<string | null, PlatformError | JsonError>

  readonly readMeta: (
    sessionId: string
  ) => Effect.Effect<StoredSessionMeta | null, PlatformError | JsonError>

  readonly writeMeta: (
    sessionId: string,
    meta: StoredSessionMeta
  ) => Effect.Effect<void, PlatformError | JsonError>

  readonly updateMeta: (
    sessionId: string,
    updater: (current: StoredSessionMeta | null) => StoredSessionMeta
  ) => Effect.Effect<StoredSessionMeta, PlatformError | JsonError>

  readonly deleteSession: (
    sessionId: string
  ) => Effect.Effect<void, PlatformError | JsonError>

  readonly readCwdIndex: (
    cwd: string
  ) => Effect.Effect<CwdIndex | null, PlatformError | JsonError>

  readonly writeCwdIndex: (
    cwd: string,
    sessionIds: readonly string[]
  ) => Effect.Effect<void, PlatformError | JsonError>

  readonly readEvents: <T>(
    sessionId: string
  ) => Effect.Effect<T[], PlatformError | JsonLinesError>

  readonly readEventsAfterCursor: <T extends { readonly timestamp: number }>(
    sessionId: string,
    cursor: StoredEventCursor
  ) => Effect.Effect<T[] | null, PlatformError | JsonLinesError>

  readonly appendEvents: <T>(
    sessionId: string,
    events: readonly T[]
  ) => Effect.Effect<void, PlatformError | JsonLinesError>

  readonly appendEventsWithCursor: <T extends { readonly timestamp: number }>(
    sessionId: string,
    events: readonly T[]
  ) => Effect.Effect<StoredEventCursor | null, PlatformError | JsonLinesError>

  readonly readProjectionSnapshot: (
    sessionId: string
  ) => Effect.Effect<unknown | null, PlatformError | JsonError>

  readonly writeProjectionSnapshot: <A>(
    sessionId: string,
    envelope: A
  ) => Effect.Effect<void, PlatformError | JsonError>

  readonly readAddressedEntry: (
    sessionId: string,
    namespace: string,
    address: string
  ) => Effect.Effect<StoredAddressedEntry | null, PlatformError | JsonError>

  readonly statAddressedEntry: (
    sessionId: string,
    namespace: string,
    address: string
  ) => Effect.Effect<StoredAddressedEntryStats | null, PlatformError>

  readonly writeAddressedEntry: (
    sessionId: string,
    namespace: string,
    address: string,
    value: unknown
  ) => Effect.Effect<void, PlatformError | JsonError>

  readonly readEventsFromPath: <T>(
    eventsPath: string
  ) => Effect.Effect<T[], PlatformError | JsonLinesError>

  readonly appendLogs: <T>(
    sessionId: string,
    entries: readonly T[]
  ) => Effect.Effect<void, PlatformError | JsonLinesError>

  readonly clearLog: (
    sessionId: string
  ) => Effect.Effect<void, PlatformError>

  readonly createSessionScratchpad: (
    sessionId: string,
  ) => Effect.Effect<string, PlatformError>

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
  ) => Effect.Effect<string, PlatformError | JsonError>

  readonly listPendingMemoryJobFiles: () => Effect.Effect<string[], PlatformError | JsonError>

  readonly listPendingMemoryJobIds: () => Effect.Effect<string[], PlatformError | JsonError>

  readonly readPendingMemoryJob: (
    input: { readonly jobId: string } | { readonly filePath: string }
  ) => Effect.Effect<MemoryExtractionJobRecord, PlatformError | JsonError>

  readonly markPendingMemoryJobRunning: (
    input: { readonly jobId: string } | { readonly filePath: string },
    job?: MemoryExtractionJobRecord
  ) => Effect.Effect<MemoryExtractionJobRecord, PlatformError | JsonError>

  readonly markPendingMemoryJobPending: (
    input: { readonly jobId: string } | { readonly filePath: string },
    job?: MemoryExtractionJobRecord
  ) => Effect.Effect<MemoryExtractionJobRecord, PlatformError | JsonError>

  readonly removePendingMemoryJob: (
    input: { readonly jobId: string } | { readonly filePath: string }
  ) => Effect.Effect<void, PlatformError>

  readonly resolvePendingMemoryJobPath: (
    jobId: string
  ) => string
}

export class SessionStorage extends Context.Tag('SessionStorage')<
  SessionStorage,
  SessionStorageShape
>() {}
