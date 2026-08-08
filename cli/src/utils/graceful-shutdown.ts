import type { CliRenderer } from '@opentui/core'
import { logger } from '@magnitudedev/logger'

let cleanupRan = false

/**
 * Reset terminal state by writing escape sequences.
 * This ensures the terminal is left in a usable state after exit.
 */
export function restoreTerminalState() {
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

async function performCleanupAndExit(
  renderer: CliRenderer,
  exitCode: number,
  beforeExit?: () => Promise<void>,
  afterCleanup?: () => void,
) {
  if (cleanupRan) return
  cleanupRan = true

  if (beforeExit) {
    try {
      await beforeExit()
    } catch {}
  }

  restoreTerminalState()
  renderer.destroy()
  afterCleanup?.()
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
export function installGracefulShutdownHandlers(
  renderer: CliRenderer,
  beforeExit?: () => Promise<void>,
  afterCleanup?: () => void,
) {
  const cleanup = (code: number) => () => { performCleanupAndExit(renderer, code, beforeExit, afterCleanup) }
  const onSigterm = cleanup(0)
  const onSighup = cleanup(0)
  const onSigint = cleanup(0)
  const onBeforeExit = cleanup(0)
  const onExit = () => {
    // On exit, we can only do synchronous cleanup
    if (!cleanupRan) {
      cleanupRan = true
      restoreTerminalState()
    }
  }
  const onUncaughtException = (err: unknown) => {
    handleCrashAndExit(renderer, 'Uncaught exception', err)
  }
  const onUnhandledRejection = (reason: unknown) => {
    handleCrashAndExit(renderer, 'Unhandled rejection', reason)
  }

  process.on('SIGTERM', onSigterm)
  process.on('SIGHUP', onSighup)
  process.on('SIGINT', onSigint)
  process.on('beforeExit', onBeforeExit)
  process.on('exit', onExit)
  process.on('uncaughtException', onUncaughtException)
  process.on('unhandledRejection', onUnhandledRejection)

  return () => {
    process.off('SIGTERM', onSigterm)
    process.off('SIGHUP', onSighup)
    process.off('SIGINT', onSigint)
    process.off('beforeExit', onBeforeExit)
    process.off('exit', onExit)
    process.off('uncaughtException', onUncaughtException)
    process.off('unhandledRejection', onUnhandledRejection)
  }
}
