import type { CliRenderer } from '@opentui/core'
import { logger } from '@magnitudedev/logger'
import { trackSessionEnd, shutdownTelemetry } from '@magnitudedev/telemetry'
import { getSessionTracker } from './telemetry-state'

let cleanupRan = false

/**
 * Reset terminal state by writing escape sequences.
 * This ensures the terminal is left in a usable state after exit.
 */
function restoreTerminalState() {
  const sequences = [
    '\x1b[?1000l', // Disable X10 mouse mode
    '\x1b[?1002l', // Disable button event mouse mode
    '\x1b[?1003l', // Disable any-event mouse mode
    '\x1b[?1006l', // Disable SGR extended mouse mode
    '\x1b[?1004l', // Disable focus reporting
    '\x1b[?2004l', // Disable bracketed paste mode
    '\x1b[?25h',   // Show cursor
  ]
  process.stdout.write(sequences.join(''))
}

async function performCleanupAndExit(renderer: CliRenderer, exitCode: number) {
  if (cleanupRan) return
  cleanupRan = true

  // Emit session_end telemetry and flush PostHog
  const tracker = getSessionTracker()
  if (tracker) {
    trackSessionEnd(tracker.getSummary())
  }
  await shutdownTelemetry()

  restoreTerminalState()
  renderer.destroy()
  process.exit(exitCode)
}

function handleCrashAndExit(renderer: CliRenderer, label: string, err: unknown) {
  if (cleanupRan) return
  cleanupRan = true

  const isError = err instanceof Error
  const message = isError ? err.message : String(err)
  const stack = isError ? err.stack : undefined

  // Log to file
  logger.error({ error: message, stack }, label)

  // Tear down TUI so stderr is visible
  restoreTerminalState()
  renderer.destroy()

  // Print to stderr now that the terminal is restored
  process.stderr.write(`\n${label}: ${message}\n`)
  if (stack) {
    process.stderr.write(stack + '\n')
  }

  process.exit(1)
}

/**
 * Install process-level signal handlers for graceful cleanup.
 * Handles SIGINT, SIGTERM, SIGHUP, and various exit scenarios.
 */
export function installGracefulShutdownHandlers(renderer: CliRenderer) {
  const cleanup = (code: number) => () => { performCleanupAndExit(renderer, code) }

  process.on('SIGTERM', cleanup(0))
  process.on('SIGHUP', cleanup(0))
  process.on('SIGINT', cleanup(0))
  process.on('beforeExit', cleanup(0))
  process.on('exit', () => {
    // On exit, we can only do synchronous cleanup
    if (!cleanupRan) {
      cleanupRan = true
      restoreTerminalState()
    }
  })
  process.on('uncaughtException', (err) => {
    handleCrashAndExit(renderer, 'Uncaught exception', err)
  })
  process.on('unhandledRejection', (reason) => {
    handleCrashAndExit(renderer, 'Unhandled rejection', reason)
  })
}