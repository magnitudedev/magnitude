import { mkdirSync, existsSync, unlinkSync, appendFileSync } from 'fs'
import { join, dirname } from 'path'
import { homedir } from 'os'

// =============================================================================
// Log File Configuration
// =============================================================================

const GLOBAL_LOG_DIR = join(homedir(), '.magnitude', 'logs')
const GLOBAL_LOG_FILE = join(GLOBAL_LOG_DIR, 'cli.jsonl')
const GLOBAL_EVENT_LOG_FILE = join(GLOBAL_LOG_DIR, 'events.jsonl')

// Ensure global log directory exists
mkdirSync(GLOBAL_LOG_DIR, { recursive: true })

// Session-specific log configuration
let sessionLogDir: string | null = null
let sessionLogFile: string | null = null

/**
 * Configure logger to write to a session-specific directory
 */
export function configureSessionLogging(sessionDir: string): void {
  sessionLogDir = sessionDir
  sessionLogFile = join(sessionDir, 'logs.jsonl')
  
  // Ensure session directory exists
  mkdirSync(sessionDir, { recursive: true })
}

/**
 * Reset logger to use global log files
 */
export function resetToGlobalLogging(): void {
  sessionLogDir = null
  sessionLogFile = null
}

/**
 * Get the current log file path (session-specific or global)
 */
export function getLogPath(): string {
  return sessionLogFile ?? GLOBAL_LOG_FILE
}

/**
 * Get the global event log path (events are not session-specific in logger)
 */
export function getEventLogPath(): string {
  return GLOBAL_EVENT_LOG_FILE
}

export function clearLog(): void {
  const logPath = getLogPath()
  try {
    if (existsSync(logPath)) {
      unlinkSync(logPath)
    }
  } catch (error) {
    // Ignore errors
  }
}

export function clearEventLog(): void {
  try {
    if (existsSync(GLOBAL_EVENT_LOG_FILE)) {
      unlinkSync(GLOBAL_EVENT_LOG_FILE)
    }
  } catch (error) {
    // Ignore errors
  }
}

// =============================================================================
// Logger Implementation
// =============================================================================

type LogLevel = 'DEBUG' | 'INFO' | 'WARN' | 'ERROR'

function writeLog(level: LogLevel, data: Record<string, unknown> | string, message?: string): void {
  const entry: Record<string, unknown> = {
    level,
    timestamp: new Date().toISOString(),
  }

  if (typeof data === 'string') {
    entry.msg = data
  } else {
    Object.assign(entry, data)
    if (message) {
      entry.msg = message
    }
  }

  const logPath = getLogPath()
  
  // Ensure directory exists before writing
  const logDir = dirname(logPath)
  if (!existsSync(logDir)) {
    mkdirSync(logDir, { recursive: true })
  }
  
  try {
    appendFileSync(logPath, JSON.stringify(entry) + '\n')
  } catch (error) {
    // Ignore write errors
  }

  for (const sub of logSubscribers) {
    sub(entry as LogEntry)
  }
}

// =============================================================================
// Log Subscribers (for debug panel live view)
// =============================================================================

export interface LogEntry {
  readonly level: LogLevel
  readonly timestamp: string
  readonly msg?: string
  readonly [key: string]: unknown
}

type LogSubscriber = (entry: LogEntry) => void
const logSubscribers: LogSubscriber[] = []

export function subscribeToLogs(callback: LogSubscriber): () => void {
  logSubscribers.push(callback)
  return () => {
    const idx = logSubscribers.indexOf(callback)
    if (idx >= 0) logSubscribers.splice(idx, 1)
  }
}

// =============================================================================
// Logger API (pino-style)
// =============================================================================

export const logger = {
  debug: (data: Record<string, unknown> | string, message?: string) => writeLog('DEBUG', data, message),
  info: (data: Record<string, unknown> | string, message?: string) => writeLog('INFO', data, message),
  warn: (data: Record<string, unknown> | string, message?: string) => writeLog('WARN', data, message),
  error: (data: Record<string, unknown> | string, message?: string) => writeLog('ERROR', data, message),
}

// =============================================================================
// Event Logger
// =============================================================================

export function logEvent(event: object & { readonly type: string }): void {
  const entry = {
    timestamp: new Date().toISOString(),
    ...event,
  }

  try {
    appendFileSync(GLOBAL_EVENT_LOG_FILE, JSON.stringify(entry) + '\n')
  } catch (error) {
    // Ignore write errors
  }
}
