import { appendJsonLines, ensureDir, readJsonFile, writeJsonFile } from '../io'
import {
  appendJsonLinesSync,
  readJsonFileSync,
  writeJsonFileSync,
} from '../io/sync'
import type { GlobalStoragePaths } from '../paths/global-paths'
import type { StoredTraceSessionMeta } from '../types/trace'

export async function initTraceSession<T extends Record<string, unknown>>(
  paths: GlobalStoragePaths,
  traceId: string,
  meta: T
): Promise<void> {
  await ensureDir(paths.traceDir(traceId))
  await writeJsonFile(paths.traceMetaFile(traceId), meta)
}

export async function appendTraces<T extends Record<string, unknown>>(
  paths: GlobalStoragePaths,
  traceId: string,
  traces: readonly T[]
): Promise<void> {
  await ensureDir(paths.traceDir(traceId))
  await appendJsonLines(paths.traceEventsFile(traceId), traces)
}

export async function readTraceMeta<
  T extends Record<string, unknown> = StoredTraceSessionMeta,
>(
  paths: GlobalStoragePaths,
  traceId: string
): Promise<T | null> {
  return readJsonFile<T | null>(paths.traceMetaFile(traceId), { fallback: null })
}

export async function writeTraceMeta<T extends Record<string, unknown>>(
  paths: GlobalStoragePaths,
  traceId: string,
  meta: T
): Promise<void> {
  await ensureDir(paths.traceDir(traceId))
  await writeJsonFile(paths.traceMetaFile(traceId), meta)
}

export async function updateTraceMeta<
  T extends Record<string, unknown> = Record<string, unknown>,
>(
  paths: GlobalStoragePaths,
  traceId: string,
  updater: (current: T | null) => T
): Promise<T> {
  const current = await readTraceMeta<T>(paths, traceId)
  const next = updater(current)
  await writeTraceMeta(paths, traceId, next)
  return next
}

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

export function initTraceSessionSync<T extends Record<string, unknown>>(
  paths: GlobalStoragePaths,
  traceId: string,
  meta: T
): void {
  writeJsonFileSync(paths.traceMetaFile(traceId), meta)
}

export function appendTracesSync<T extends Record<string, unknown>>(
  paths: GlobalStoragePaths,
  traceId: string,
  traces: readonly T[]
): void {
  if (traces.length === 0) {
    return
  }

  appendJsonLinesSync(paths.traceEventsFile(traceId), traces)
}

export function readTraceMetaSync<
  T extends Record<string, unknown> = StoredTraceSessionMeta,
>(paths: GlobalStoragePaths, traceId: string): T | null {
  return readJsonFileSync<T | null>(paths.traceMetaFile(traceId), null)
}

export function writeTraceMetaSync<T extends Record<string, unknown>>(
  paths: GlobalStoragePaths,
  traceId: string,
  meta: T
): void {
  writeJsonFileSync(paths.traceMetaFile(traceId), meta)
}

export function updateTraceMetaSync<
  T extends Record<string, unknown> = Record<string, unknown>,
>(
  paths: GlobalStoragePaths,
  traceId: string,
  updater: (current: T | null) => T
): T {
  const current = readTraceMetaSync<T>(paths, traceId)
  const next = updater(current)
  writeTraceMetaSync(paths, traceId, next)
  return next
}