import type { PlatformError } from '@effect/platform/Error'
import { Context, Effect } from 'effect'

import type { JsonLinesError } from '../io/storage'
import type { StoredLogEntry } from '../types'

export interface LogStorageShape {
  readonly appendSession: (
    sessionId: string,
    entries: readonly StoredLogEntry[]
  ) => Effect.Effect<void, PlatformError | JsonLinesError>

  readonly clearSession: (sessionId: string) => Effect.Effect<void, PlatformError>

  readonly getSessionPath: (sessionId: string) => string
}

export class LogStorage extends Context.Tag('LogStorage')<
  LogStorage,
  LogStorageShape
>() {}
