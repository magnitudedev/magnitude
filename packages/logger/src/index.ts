// Browser-safe logger — spec §6.5
//
// In Node (CLI/ACN), logs are persisted to disk via @magnitudedev/storage.
// In the browser (web/desktop renderer), we skip disk persistence entirely
// and only emit to console + in-memory subscribers.
//
// The key constraint: @magnitudedev/storage imports node:os, node:path,
// node:crypto — all Node-only. We must NOT import it at module load time
// in browser contexts.

import type { GlobalStoragePaths } from '@magnitudedev/storage'

// Detect browser: window exists, process is undefined or is the Vite stub
const isBrowser =
  typeof globalThis !== 'undefined' &&
  typeof (globalThis as unknown as { window?: unknown }).window !== 'undefined'

// Storage module reference — lazily loaded in Node
let storageModule: typeof import('@magnitudedev/storage') | null = null

// Eagerly load storage in Node environments
if (!isBrowser) {
  try {
    // Use dynamic import — works in both Bun and Node ESM
    // The await is safe because Bun/Node support top-level await in ESM
    storageModule = await import('@magnitudedev/storage')
  } catch {
    // Storage not available — logger works in console-only mode
  }
}

let globalPaths: GlobalStoragePaths | null = null
if (storageModule) {
  try {
    globalPaths = storageModule.makeGlobalStoragePaths(storageModule.defaultGlobalStorageRoot())
  } catch {
    // ignore
  }
}

type LogLevel = 'DEBUG' | 'INFO' | 'WARN' | 'ERROR'

export interface LogEntry {
  readonly level: LogLevel
  readonly timestamp: string
  readonly msg?: string
  readonly [key: string]: unknown
}

type LogSubscriber = (entry: LogEntry) => void
const logSubscribers: LogSubscriber[] = []

let activeSessionId: string | null = null
let activePaths: GlobalStoragePaths | null = null
let warnedAboutNoInit = false

export interface Logger {
  debug(data: Record<string, unknown> | string, message?: string): void
  info(data: Record<string, unknown> | string, message?: string): void
  warn(data: Record<string, unknown> | string, message?: string): void
  error(data: Record<string, unknown> | string, message?: string): void
}

function writeLog(
  level: LogLevel,
  data: Record<string, unknown> | string,
  message?: string,
  sessionId?: string,
  paths?: GlobalStoragePaths
): void {
  const timestamp = new Date().toISOString()
  const entry: LogEntry =
    typeof data === 'string'
      ? {
          level,
          timestamp,
          msg: data,
        }
      : {
          level,
          timestamp,
          ...data,
          ...(message ? { msg: message } : {}),
        }

  if (sessionId && paths && storageModule) {
    try {
      storageModule.appendSessionLogsSync(paths, sessionId, [entry])
    } catch {
      // Ignore write errors
    }
  } else if (!activeSessionId) {
    if (!warnedAboutNoInit) {
      console.error('[logger] Warning: logger used before initLogger(sessionId) — logs will not be persisted to disk')
      warnedAboutNoInit = true
    }
  }

  for (const sub of logSubscribers) {
    sub(entry)
  }
}

export function initLogger(sessionId: string): void {
  if (isBrowser || !storageModule) return
  activeSessionId = sessionId
  try {
    activePaths = storageModule.makeGlobalStoragePaths(storageModule.defaultGlobalStorageRoot())
  } catch {
    // ignore
  }
}

export const logger: Logger = {
  debug: (data, message?) =>
    writeLog('DEBUG', data, message, activeSessionId ?? undefined, activePaths ?? undefined),
  info: (data, message?) =>
    writeLog('INFO', data, message, activeSessionId ?? undefined, activePaths ?? undefined),
  warn: (data, message?) =>
    writeLog('WARN', data, message, activeSessionId ?? undefined, activePaths ?? undefined),
  error: (data, message?) =>
    writeLog('ERROR', data, message, activeSessionId ?? undefined, activePaths ?? undefined),
}

export function createLogger(sessionId: string): Logger {
  if (isBrowser || !storageModule) {
    return logger
  }
  let loggerPaths: GlobalStoragePaths | null = null
  try {
    loggerPaths = storageModule.makeGlobalStoragePaths(storageModule.defaultGlobalStorageRoot())
  } catch {
    // ignore
  }

  return {
    debug: (data, message?) => writeLog('DEBUG', data, message, sessionId, loggerPaths ?? undefined),
    info: (data, message?) => writeLog('INFO', data, message, sessionId, loggerPaths ?? undefined),
    warn: (data, message?) => writeLog('WARN', data, message, sessionId, loggerPaths ?? undefined),
    error: (data, message?) => writeLog('ERROR', data, message, sessionId, loggerPaths ?? undefined),
  }
}

export function subscribeToLogs(callback: LogSubscriber): () => void {
  logSubscribers.push(callback)
  return () => {
    const idx = logSubscribers.indexOf(callback)
    if (idx >= 0) logSubscribers.splice(idx, 1)
  }
}

export function clearSessionLog(sessionId: string): void {
  if (isBrowser || !globalPaths || !storageModule) return
  try {
    storageModule.clearSessionLogSync(globalPaths, sessionId)
  } catch {
    // Ignore errors
  }
}

export function getSessionLogPath(sessionId: string): string {
  if (isBrowser || !globalPaths || !storageModule) return ''
  try {
    return storageModule.getSessionLogPath(globalPaths, sessionId)
  } catch {
    return ''
  }
}
