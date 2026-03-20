import { Effect, Layer } from 'effect'

import { SessionStorage } from './contracts'
import {
  appendSessionEvents,
  createMemoryExtractionJobRecord,
  createTimestampSessionId,
  findLatestSessionId,
  listPendingMemoryJobFiles,
  listPendingMemoryJobIds,
  listSessionIds,
  markPendingMemoryJobPending,
  markPendingMemoryJobRunning,
  readPendingMemoryJob,
  readSessionEvents,
  readSessionEventsFromPath,
  readSessionMeta,
  removePendingMemoryJob,
  resolvePendingMemoryJobPath,
  writePendingMemoryJob,
  writeSessionMeta,
  createSessionWorkspace,
  updateSessionMeta,
} from './storage'
import { appendSessionLogs, clearSessionLog } from '../logs/storage'
import type { StoredLogEntry } from '../types/log'
import { GlobalStorage } from '../services'

export const SessionStorageLive = Layer.effect(
  SessionStorage,
  Effect.gen(function* () {
    const globalStorage = yield* GlobalStorage

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
        Effect.promise(() => readSessionMeta(globalStorage.paths, sessionId)),

      writeMeta: (sessionId, meta) =>
        Effect.promise(() => writeSessionMeta(globalStorage.paths, sessionId, meta)),

      updateMeta: (sessionId, updater) =>
        Effect.promise(() => updateSessionMeta(globalStorage.paths, sessionId, updater)),

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