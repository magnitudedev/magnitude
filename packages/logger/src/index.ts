import {
  appendSessionLogsSync,
  clearSessionLogSync,
  defaultGlobalStorageRoot,
  getSessionLogPath as getSessionLogPathHelper,
  makeGlobalStoragePaths,
} from '@magnitudedev/storage'
import type { GlobalStoragePaths } from '@magnitudedev/storage'

const globalPaths = makeGlobalStoragePaths(defaultGlobalStorageRoot())

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

  if (sessionId && paths) {
    try {
      appendSessionLogsSync(paths, sessionId, [entry])
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
  activeSessionId = sessionId
  activePaths = makeGlobalStoragePaths(defaultGlobalStorageRoot())
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
  const loggerPaths = makeGlobalStoragePaths(defaultGlobalStorageRoot())

  return {
    debug: (data, message?) => writeLog('DEBUG', data, message, sessionId, loggerPaths),
    info: (data, message?) => writeLog('INFO', data, message, sessionId, loggerPaths),
    warn: (data, message?) => writeLog('WARN', data, message, sessionId, loggerPaths),
    error: (data, message?) => writeLog('ERROR', data, message, sessionId, loggerPaths),
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
  try {
    clearSessionLogSync(globalPaths, sessionId)
  } catch {
    // Ignore errors
  }
}

export function getSessionLogPath(sessionId: string): string {
  return getSessionLogPathHelper(globalPaths, sessionId)
}