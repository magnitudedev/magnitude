import { appendJsonLines, ensureDir, removeFileIfExists } from '../io'
import { appendJsonLinesSync, clearFileSync } from '../io/sync'
import type { GlobalStoragePaths } from '../paths/global-paths'
import type { StoredLogEntry } from '../types/log'

export async function appendSessionLogs(
  paths: GlobalStoragePaths,
  sessionId: string,
  entries: readonly StoredLogEntry[]
): Promise<void> {
  await ensureDir(paths.sessionDir(sessionId))
  await appendJsonLines(paths.sessionLogFile(sessionId), entries)
}

export async function clearSessionLog(
  paths: GlobalStoragePaths,
  sessionId: string
): Promise<void> {
  await removeFileIfExists(paths.sessionLogFile(sessionId))
}

export function getSessionLogPath(
  paths: GlobalStoragePaths,
  sessionId: string
): string {
  return paths.sessionLogFile(sessionId)
}

export function appendSessionLogsSync(
  paths: GlobalStoragePaths,
  sessionId: string,
  entries: readonly StoredLogEntry[]
): void {
  if (entries.length === 0) {
    return
  }

  appendJsonLinesSync(paths.sessionLogFile(sessionId), entries)
}

export function clearSessionLogSync(
  paths: GlobalStoragePaths,
  sessionId: string
): void {
  clearFileSync(paths.sessionLogFile(sessionId))
}