import { Effect, Layer, Schema } from 'effect'

import { SessionStorage } from './contracts'
import {
  appendSessionEvents,
  createMemoryExtractionJobRecord,
  createSessionWorkspace,
  createTimestampSessionId,
  findLatestSessionId,
  listPendingMemoryJobFiles,
  listPendingMemoryJobIds,
  listSessionIds,
  markPendingMemoryJobPending,
  markPendingMemoryJobRunning,
  readPendingMemoryJob,
  readRawSessionMeta,
  readSessionEvents,
  readSessionEventsFromPath,
  removePendingMemoryJob,
  resolvePendingMemoryJobPath,
  updateSessionMeta,
  writePendingMemoryJob,
  writeSessionMeta,
} from './storage'
import { appendSessionLogs, clearSessionLog } from '../logs/storage'
import { GlobalStorage } from '../services'
import { Version, VersionLive } from '../services/version'
import type { StoredLogEntry } from '../types/log'
import { StoredSessionMetaSchema, type StoredSessionMeta } from '../types/session'

export function SessionStorageLive() {
  return Layer.effect(
    SessionStorage,
    Effect.gen(function* () {
      const globalStorage = yield* GlobalStorage
      const version = yield* Version
      const versionLayer = VersionLive(version.getVersion())

      const decodeStoredSessionMeta = (
        raw: unknown
      ): Effect.Effect<StoredSessionMeta> =>
        Schema.decodeUnknown(StoredSessionMetaSchema)(raw).pipe(
          Effect.provide(versionLayer),
          Effect.orDie,
        )

      return SessionStorage.of({
        paths: {
          root: globalStorage.paths.root,
          sessionsRoot: globalStorage.paths.sessionsRoot,
          pendingMemoryExtractionRoot: globalStorage.paths.pendingMemoryExtractionRoot,
          sessionDir: globalStorage.paths.sessionDir,
          sessionMetaFile: globalStorage.paths.sessionMetaFile,
          sessionEventsFile: globalStorage.paths.sessionEventsFile,
          sessionLogFile: globalStorage.paths.sessionLogFile,
          sessionWorkspace: globalStorage.paths.sessionWorkspace,
          pendingMemoryJobFile: globalStorage.paths.pendingMemoryJobFile,
        },

        createTimestampSessionId,

        listSessionIds: (options) =>
          Effect.promise(() => listSessionIds(globalStorage.paths, options)),

        findLatestSessionId: (options) =>
          Effect.promise(() => findLatestSessionId(globalStorage.paths, options)),

        readMeta: (sessionId) =>
          Effect.gen(function* () {
            const raw = yield* Effect.promise(() => readRawSessionMeta(globalStorage.paths, sessionId))
            if (raw === null) {
              return null
            }
            return yield* decodeStoredSessionMeta(raw)
          }),

        writeMeta: (sessionId, meta) =>
          Effect.promise(() => writeSessionMeta(globalStorage.paths, sessionId, meta)),

        updateMeta: (sessionId, updater) =>
          Effect.gen(function* () {
            const raw = yield* Effect.promise(() => readRawSessionMeta(globalStorage.paths, sessionId))
            const current = raw === null ? null : yield* decodeStoredSessionMeta(raw)
            return yield* Effect.promise(() =>
              updateSessionMeta(globalStorage.paths, sessionId, current, updater)
            )
          }),

        readEvents: <T>(sessionId: string) =>
          Effect.promise(() => readSessionEvents<T>(globalStorage.paths, sessionId)),

        appendEvents: <T>(sessionId: string, events: readonly T[]) =>
          Effect.promise(() => appendSessionEvents(globalStorage.paths, sessionId, events)),

        readEventsFromPath: <T>(eventsPath: string) =>
          Effect.promise(() => readSessionEventsFromPath<T>(eventsPath)),

        appendLogs: <T>(sessionId: string, entries: readonly T[]) =>
          Effect.promise(() =>
            appendSessionLogs(globalStorage.paths, sessionId, entries as readonly StoredLogEntry[])
          ),

        clearLog: (sessionId) =>
          Effect.promise(() =>
            clearSessionLog(globalStorage.paths, sessionId)
          ),

        createSessionWorkspace: (sessionId, cwd) =>
          Effect.promise(() =>
            createSessionWorkspace(globalStorage.paths, sessionId, cwd)
          ),

        createMemoryExtractionJobRecord,

        writePendingMemoryJob: (job) =>
          Effect.promise(() => writePendingMemoryJob(globalStorage.paths, job)),

        listPendingMemoryJobFiles: () =>
          Effect.promise(() => listPendingMemoryJobFiles(globalStorage.paths)),

        listPendingMemoryJobIds: () =>
          Effect.promise(() => listPendingMemoryJobIds(globalStorage.paths)),

        readPendingMemoryJob: (input) =>
          Effect.promise(() => readPendingMemoryJob(globalStorage.paths, input)),

        markPendingMemoryJobRunning: (input, job) =>
          Effect.promise(() => markPendingMemoryJobRunning(globalStorage.paths, input, job)),

        markPendingMemoryJobPending: (input, job) =>
          Effect.promise(() => markPendingMemoryJobPending(globalStorage.paths, input, job)),

        removePendingMemoryJob: (input) =>
          Effect.promise(() => removePendingMemoryJob(globalStorage.paths, input)),

        resolvePendingMemoryJobPath: (jobId) =>
          resolvePendingMemoryJobPath(globalStorage.paths, jobId),
      })
  })
  )
}