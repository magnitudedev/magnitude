import {
  appendJsonLinesSync,
  readJsonFileSync,
  writeJsonFileSync,
} from '../io/sync'
import type { GlobalStoragePaths } from '../paths/global-paths'
import type { StoredTraceSessionMeta } from '../types/trace'

export function getTraceDir(paths: GlobalStoragePaths, traceId: string): string {
  return paths.traceDir(traceId)
}

export function getTraceMetaPath(
  paths: GlobalStoragePaths,
  traceId: string
): string {
  return paths.traceMetaFile(traceId)
}

export function getTraceEventsPath(
  paths: GlobalStoragePaths,
  traceId: string
): string {
  return paths.traceEventsFile(traceId)
}

export function initTraceSessionSync(
  paths: GlobalStoragePaths,
  traceId: string,
  meta: StoredTraceSessionMeta
): void {
  writeJsonFileSync(paths.traceMetaFile(traceId), meta)
}

export function appendTracesSync(
  paths: GlobalStoragePaths,
  traceId: string,
  traces: readonly unknown[]
): void {
  if (traces.length === 0) {
    return
  }

  appendJsonLinesSync(paths.traceEventsFile(traceId), traces)
}

export function readTraceMetaSync(
  paths: GlobalStoragePaths,
  traceId: string
): StoredTraceSessionMeta | null {
  return readJsonFileSync<StoredTraceSessionMeta | null>(
    paths.traceMetaFile(traceId),
    null
  )
}

export function writeTraceMetaSync(
  paths: GlobalStoragePaths,
  traceId: string,
  meta: StoredTraceSessionMeta
): void {
  writeJsonFileSync(paths.traceMetaFile(traceId), meta)
}

export function updateTraceMetaSync(
  paths: GlobalStoragePaths,
  traceId: string,
  updater: (current: StoredTraceSessionMeta | null) => StoredTraceSessionMeta
): StoredTraceSessionMeta {
  const current = readTraceMetaSync(paths, traceId)
  const next = updater(current)
  writeTraceMetaSync(paths, traceId, next)
  return next
}
