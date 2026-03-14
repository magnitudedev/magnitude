import { Effect, Layer } from 'effect'

import { LogStorage } from './contracts'
import {
  appendSessionLogs,
  clearSessionLog,
  getSessionLogPath,
} from './storage'
import { GlobalStorage } from '../services'
import type { StoredLogEntry } from '../types'

export const LogStorageLive = Layer.effect(
  LogStorage,
  Effect.gen(function* () {
    const globalStorage = yield* GlobalStorage

    return LogStorage.of({
      appendSession: (
        sessionId: string,
        entries: readonly StoredLogEntry[]
      ) =>
        Effect.promise(() =>
          appendSessionLogs(globalStorage.paths, sessionId, entries)
        ),

      clearSession: (sessionId: string) =>
        Effect.promise(() => clearSessionLog(globalStorage.paths, sessionId)),

      getSessionPath: (sessionId: string) =>
        getSessionLogPath(globalStorage.paths, sessionId),
    })
  })
)