import { Context, Effect } from 'effect'

import type { StoredLogEntry } from '../types'

export interface LogStorageShape {
  readonly appendSession: (
    sessionId: string,
    entries: readonly StoredLogEntry[]
  ) => Effect.Effect<void>

  readonly clearSession: (sessionId: string) => Effect.Effect<void>

  readonly getSessionPath: (sessionId: string) => string
}

export class LogStorage extends Context.Tag('LogStorage')<
  LogStorage,
  LogStorageShape
>() {}