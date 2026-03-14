import { Context, Effect } from 'effect'

import type { StoredTraceSessionMeta } from '../types/trace'

export interface TraceStorageShape {
  readonly initSession: <T extends Record<string, unknown>>(
    traceId: string,
    meta: T
  ) => Effect.Effect<void>

  readonly append: <T extends Record<string, unknown>>(
    traceId: string,
    traces: readonly T[]
  ) => Effect.Effect<void>

  readonly readMeta: <
    T extends Record<string, unknown> = StoredTraceSessionMeta,
  >(
    traceId: string
  ) => Effect.Effect<T | null>

  readonly writeMeta: <T extends Record<string, unknown>>(
    traceId: string,
    meta: T
  ) => Effect.Effect<void>

  readonly updateMeta: <
    T extends Record<string, unknown> = Record<string, unknown>,
  >(
    traceId: string,
    updater: (current: T | null) => T
  ) => Effect.Effect<T>

  readonly getDirPath: (traceId: string) => string
  readonly getMetaPath: (traceId: string) => string
  readonly getEventsPath: (traceId: string) => string
}

export class TraceStorage extends Context.Tag('TraceStorage')<
  TraceStorage,
  TraceStorageShape
>() {}