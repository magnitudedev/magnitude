/**
 * Effect Logger → file logger bridge.
 *
 * Effect's default logger writes to stderr, which corrupts the TUI rendering.
 * This layer replaces it with one that forwards to @magnitudedev/logger
 * (which writes to the active session's logs.jsonl). Nothing hits stderr.
 */

import { Cause, HashMap, Layer, Logger } from 'effect'
import { logger } from '@magnitudedev/logger'

const fileLogger = Logger.make(({ logLevel, message, annotations, cause }) => {
  const data: Record<string, unknown> = {}

  // HashMap → plain object for the structured log entry.
  for (const [k, v] of HashMap.entries(annotations)) {
    data[k] = v
  }

  // Include cause when present (Effect distinguishes "no cause" via Cause.isEmpty).
  if (!Cause.isEmpty(cause)) {
    data.cause = Cause.pretty(cause)
  }

  // Effect's `message` is `unknown`; coerce to a string. Arrays show up when
  // multiple args are passed to Effect.log (e.g. Effect.logError(msg, cause)).
  const msg = Array.isArray(message)
    ? message.map((m) => (typeof m === 'string' ? m : JSON.stringify(m))).join(' ')
    : typeof message === 'string'
      ? message
      : JSON.stringify(message)

  switch (logLevel.label) {
    case 'FATAL':
    case 'ERROR':
      logger.error(data, msg)
      return
    case 'WARN':
      logger.warn(data, msg)
      return
    case 'INFO':
      logger.info(data, msg)
      return
    case 'DEBUG':
    case 'TRACE':
    default:
      logger.debug(data, msg)
      return
  }
})

/**
 * Layer that replaces Effect's default logger with one that writes to the
 * file logger only. Mount in the agent runtime so all Effect.log* calls
 * inside workers/projections route through @magnitudedev/logger.
 */
export const EffectLoggerLayer = Logger.replace(Logger.defaultLogger, fileLogger)
