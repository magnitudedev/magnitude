import * as FileSystem from '@effect/platform/FileSystem'
import * as Path from '@effect/platform/Path'
import { type PlatformError } from '@effect/platform/Error'
import { Effect } from 'effect'

import { makeStorageIo, type JsonLinesError } from '../io/storage'
import { GlobalStorage } from '../services'
import type { StoredLogEntry } from '../types/log'
import type { LogStorageShape } from './contracts'

export function makeLogStorage(): Effect.Effect<
  LogStorageShape,
  never,
  FileSystem.FileSystem | Path.Path | GlobalStorage
> {
  return Effect.gen(function* () {
    const io = yield* makeStorageIo()
    const globalStorage = yield* GlobalStorage
    const g = globalStorage.paths

    return {
      appendSession: (sessionId, entries) =>
        Effect.gen(function* () {
          yield* io.ensureDir(g.sessionDir(sessionId))
          yield* io.appendJsonLines(g.sessionLogFile(sessionId), entries)
        }),

      clearSession: (sessionId) =>
        io.removeFileIfExists(g.sessionLogFile(sessionId)),

      getSessionPath: (sessionId) => g.sessionLogFile(sessionId),
    }
  })
}
