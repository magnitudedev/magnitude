import { Effect, Layer } from 'effect'

import { TraceStorage } from './contracts'
import {
  appendTraces,
  getTraceDir,
  getTraceEventsPath,
  getTraceMetaPath,
  initTraceSession,
  readTraceMeta,
  updateTraceMeta,
  writeTraceMeta,
} from './storage'
import { GlobalStorage } from '../services'

export const TraceStorageLive = Layer.effect(
  TraceStorage,
  Effect.gen(function* () {
    const globalStorage = yield* GlobalStorage

    return TraceStorage.of({
      initSession: <T extends Record<string, unknown>>(
        traceId: string,
        meta: T
      ) =>
        Effect.promise(() =>
          initTraceSession(
            globalStorage.paths,
            traceId,
            meta as Parameters<typeof initTraceSession>[2]
          )
        ),
      append: <T extends Record<string, unknown>>(
        traceId: string,
        traces: readonly T[]
      ) => Effect.promise(() => appendTraces(globalStorage.paths, traceId, traces)),
      readMeta: <T extends Record<string, unknown> = Record<string, unknown>>(
        traceId: string
      ) => Effect.promise(() => readTraceMeta<T>(globalStorage.paths, traceId)),
      writeMeta: <T extends Record<string, unknown>>(
        traceId: string,
        meta: T
      ) => Effect.promise(() => writeTraceMeta(globalStorage.paths, traceId, meta)),
      updateMeta: <T extends Record<string, unknown> = Record<string, unknown>>(
        traceId: string,
        updater: (current: T | null) => T
      ) =>
        Effect.promise(() =>
          updateTraceMeta<T>(globalStorage.paths, traceId, updater)
        ),

      getDirPath: (traceId: string) => getTraceDir(globalStorage.paths, traceId),
      getMetaPath: (traceId: string) =>
        getTraceMetaPath(globalStorage.paths, traceId),
      getEventsPath: (traceId: string) =>
        getTraceEventsPath(globalStorage.paths, traceId),
    })
  })
)